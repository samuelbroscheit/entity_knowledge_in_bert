import base64
import glob
import io
import json
import math
import multiprocessing
import os
import pandas
import pickle
import numpy
import tqdm
from collections import Counter
from typing import Dict

from misc import normalize_wiki_entity, get_stopwordless_token_set
from pipeline_job import PipelineJob
from pytorch_pretrained_bert import BertTokenizer


class CreateWikiTrainingData(PipelineJob):
    """
    Create sequence labelling data by tokenizing with BertTokenizer and adding labels
    for these spans that have an associated Wikipedia link. Also, annotate spans detected
    by the keyword matcher that is aware of mentions that are linking to the entities
    in our top k popular entities dictionary.
    Subsequently, we count the mentions in this data and create a discounted prior p(e|m)
    and a set of necessary articles that contain the top k popular entities.
    """

    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                "data/indexes/redirects_en.ttl.bz2.dict",
                f"data/versions/{opts.data_version_name}/indexes/keyword_processor.pickle",
                f"data/versions/{opts.data_version_name}/indexes/popular_entity_counter_dict.pickle",
                f"data/versions/{opts.data_version_name}/indexes/mention_entity_counter_popular_entities.pickle",
                f"data/versions/{opts.data_version_name}/wikiextractor_out/{opts.wiki_lang_version}/",
            ],
            provides=[
                f"data/versions/{opts.data_version_name}/wiki_training/raw/{opts.wiki_lang_version}/",
                f"data/versions/{opts.data_version_name}/indexes/necessary_articles.pickle",
                f"data/versions/{opts.data_version_name}/indexes/entities_in_necessary_articles.pickle",
                f"data/versions/{opts.data_version_name}/indexes/mention_counter.pickle",
                f"data/versions/{opts.data_version_name}/indexes/mention_entity_discounted_probs.pickle",
                f"data/versions/{opts.data_version_name}/indexes/entities_found_in_article.pickle",
                f"data/versions/{opts.data_version_name}/indexes/article_found_for_entities.pickle",
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        with open("data/indexes/redirects_en.ttl.bz2.dict", "rb") as f:
            redirects_en = pickle.load(f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/keyword_processor.pickle", "rb",) as f:
            keyword_processor = pickle.load(f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/popular_entity_counter_dict.pickle", "rb",
        ) as f:
            most_popular_entity_counter_dict = pickle.load(f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/mention_entity_counter_popular_entities.pickle", "rb",
        ) as f:
            mention_entity_count_popular_entities = pickle.load(f)

        self.log("Loading finished")

        in_queue = multiprocessing.Queue()
        out_queue = multiprocessing.Queue()

        workers = list()

        list_dir_string = (
            f"data/versions/{self.opts.data_version_name}/wikiextractor_out/{self.opts.wiki_lang_version}/*/*/wiki_*"
        )

        #
        # start the workers in individual processes
        #

        if self.opts.uncased:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

        for id in range(self.opts.create_training_data_num_workers):
            worker = Worker(
                in_queue,
                out_queue,
                opts=self.opts,
                tokenizer=tokenizer,
                redirects_en=redirects_en,
                keyword_processor=keyword_processor,
                popular_entity_counter_dict=most_popular_entity_counter_dict,
                mention_entity_counter_popular_entities=mention_entity_count_popular_entities,
            )
            worker.start()
            workers.append(worker)

        self.log("Fill queue")
        # fill the queue
        for file_nr, extracted_wiki_file in enumerate(tqdm.tqdm(glob.glob(list_dir_string))):
            in_queue.put(extracted_wiki_file)

        outputs = list()
        mention_counter = Counter()
        entities_found_in_article = dict()
        article_found_for_entities = dict()

        self.log("Collect the output")
        # collect the output
        for file_nr, extracted_wiki_file in enumerate(tqdm.tqdm(glob.glob(list_dir_string))):
            ((out_file_names, local_mention_counter, local_entities_found_in_article,), in_file_name,) = out_queue.get()
            outputs.append((out_file_names, in_file_name))
            mention_counter.update(local_mention_counter)
            for (out_file_name, local_entities_found_counter,) in local_entities_found_in_article:
                self.debug("{} finished".format((out_file_name, in_file_name)))
                entities_found_in_article[out_file_name] = dict(local_entities_found_counter.items())
                for ent, count in local_entities_found_counter.items():
                    if ent not in article_found_for_entities:
                        article_found_for_entities[ent] = Counter()
                    article_found_for_entities[ent][out_file_name] = count

        # for file_name in outputs:
        #     print(file_name)
        #     pass

        # put the None into the queue so the loop in the run() function of the worker stops
        for worker in workers:
            in_queue.put(None)
            out_queue.put(None)

        # terminate the process
        for worker in workers:
            worker.join()

        # remove mentions for which no entities are found
        me_pop = [(k, v) for k, v in mention_entity_count_popular_entities.items() if len(v) > 0]

        # take the top 1000 mentions according to their total mention-entity count and ...
        me_topk = [k for k, c in sorted(me_pop, key=lambda x: sum([j for i, j in x[1]]))[-1000:]]

        # ... compute an idealized p(NIL|m), i.e., the probability of the most frequent mentions to
        # link to NIL. This is based on the assumption that using the most frequent mentions we can
        # find the best approximation of p(NIL|m).
        prob_m_links_to_nil = numpy.array(
            list(
                (
                        mention_counter[m_from_topk_entities]
                        - sum([c for k, c in mention_entity_count_popular_entities[m_from_topk_entities]])
                )
                / sum([c for k, c in mention_entity_count_popular_entities[m_from_topk_entities]])
                for m_from_topk_entities in me_topk
            )
        ).mean()

        # Now use the estimate of p(NIL|m) to rescale p(NIL|"United States") == 0 and p(NIL | m) for
        # other mention-entity pairs are discounted accordingly. FIrst a "hacky" version that worked
        # and then the proper version from the paper
        if self.opts.create_training_data_discount_nil_strategy == "hacky":

            me_pop_p = dict()
            for a_mention, counts in me_pop:
                sum_em_counts_for_a_mention = sum([c for k, c in mention_entity_count_popular_entities[a_mention]])
                tmp = [
                    (e, (c / (mention_counter[a_mention] + 1)))
                    for e, c in mention_entity_count_popular_entities[a_mention]
                    + [
                        (
                            "|||O|||",
                            max(
                                [
                                    math.pow(max([mention_counter[a_mention] - sum_em_counts_for_a_mention, 1,]), 3 / 4,)
                                    - math.pow(prob_m_links_to_nil * mention_counter[a_mention], 3 / 4,),
                                    1,
                                ]
                            ),
                        )
                    ]
                ]
                sum_tmp = sum([p for k, p in tmp])
                me_pop_p[a_mention] = [(k, p / sum_tmp) for k, p in tmp]

        elif self.opts.create_training_data_discount_nil_strategy == "prop":

            me_pop_p = dict()
            for a_mention, counts in me_pop:
                sum_em_counts_for_a_mention = sum([c for k, c in mention_entity_count_popular_entities[a_mention]])
                tmp = [
                    (e, (c / (mention_counter[a_mention] + 1)))
                    for e, c in mention_entity_count_popular_entities[a_mention]
                    + [
                        (
                            "|||O|||",
                            max(
                                [
                                    # == NIL count
                                    max([mention_counter[a_mention] - sum_em_counts_for_a_mention, 1,])
                                    -
                                    prob_m_links_to_nil/(1-prob_m_links_to_nil) * mention_counter[a_mention],
                                    1,
                                ]
                            ),
                        )
                    ]
                ]
                sum_tmp = sum([p for k, p in tmp])
                me_pop_p[a_mention] = [(k, p / sum_tmp) for k, p in tmp]

        # Now collect the Wikipedia articles that are neccessary to compute the entity embeddings for the most
        # popular entities, i.e. the Wikipedia articles that contain the most popular entities.
        necessary_articles = set()
        entities_in_necessary_articles = Counter()
        for out_file_name, ent_count_dict in tqdm.tqdm(
            sorted(entities_found_in_article.items(), key=lambda x: len(x[1]), reverse=True)
        ):
            for ent, count in ent_count_dict.items():
                if (
                    ent not in entities_in_necessary_articles
                    or entities_in_necessary_articles[ent]
                    < self.opts.create_training_data_num_entities_in_necessary_articles
                ):
                    entities_in_necessary_articles.update(ent_count_dict)
                    necessary_articles.add(out_file_name.replace("/tmp/", f"/{self.opts.wiki_lang_version}/"))

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/mention_entity_discounted_probs.pickle", "wb"
        ) as f:
            pickle.dump(me_pop_p, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/mention_counter.pickle", "wb") as f:
            pickle.dump(mention_counter, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/entities_found_in_article.pickle", "wb") as f:
            pickle.dump(entities_found_in_article, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/article_found_for_entities.pickle", "wb") as f:
            pickle.dump(article_found_for_entities, f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/entities_in_necessary_articles.pickle", "wb"
        ) as f:
            pickle.dump(entities_in_necessary_articles, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/necessary_articles.pickle", "wb") as f:
            pickle.dump(necessary_articles, f)

        os.rename(
            f"data/versions/{self.opts.data_version_name}/wiki_training/raw/tmp/",
            f"data/versions/{self.opts.data_version_name}/wiki_training/raw/{self.opts.wiki_lang_version}/",
        )


class Worker(multiprocessing.Process):
    def __init__(
        self,
        in_queue,
        out_queue,
        opts,
        tokenizer,
        redirects_en,
        keyword_processor,
        popular_entity_counter_dict,
        mention_entity_counter_popular_entities,
    ):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.tokenizer = tokenizer
        self.redirects_en = redirects_en
        self.keyword_processor = keyword_processor
        self.popular_entity_counter_dict = popular_entity_counter_dict
        self.mention_entity_counter_popular_entities = mention_entity_counter_popular_entities
        self.opts = opts

    def run(self):
        # this loop will run until it receives None form the in_queue, if the queue is empty
        #  the loop will wait until it gets something
        for next_item in iter(self.in_queue.get, None):
            file_name = next_item
            self.out_queue.put((self.extract_data(next_item), file_name))

    def extract_data(self, file_name):

        len_prefix = len(
            f"data/versions/{self.opts.data_version_name}/wikiextractor_out/{self.opts.wiki_lang_version}/"
        )

        local_mention_counter = Counter()
        out_file_names = list()
        local_entities_found_in_article = list()

        with io.open(file_name) as f:

            for i, wiki_article in enumerate(f.readlines()):

                wiki_article = json.loads(wiki_article)

                debug = wiki_article["id"] == "28490"
                wiki_article_title_set = get_stopwordless_token_set(wiki_article["title"])

                def corefers_with_title_entity(s):
                    s_set = get_stopwordless_token_set(s)
                    is_shorter = len(s_set) <= len(wiki_article_title_set)
                    has_overlap = len(s_set.intersection(wiki_article_title_set)) / len(wiki_article_title_set) > 0
                    return is_shorter and has_overlap

                wiki_article_normalized_wiki_entity = normalize_wiki_entity(wiki_article["title"], replace_ws=True)
                wiki_article_title_entity = self.redirects_en.get(
                    wiki_article_normalized_wiki_entity, wiki_article_normalized_wiki_entity,
                )

                local_entities_found_counter = Counter()

                start_offset_dict = dict()

                # for ((start, end), (mention, wiki_page_name)) in pickle.loads(base64.b64decode(wiki_article['internal_links'].encode('utf-8'))).items():
                #     # print(char_offsets, (mention, wiki_page_name))
                #     start_offset_dict[start] = (end, (mention, wiki_page_name))

                links_offsets = sorted(
                    pickle.loads(base64.b64decode(wiki_article["internal_links"].encode("utf-8"))).items(),
                    key=lambda x: x[0][0],
                )

                keywords_found = self.keyword_processor.extract_keywords(wiki_article["text"], span_info=True)

                title_seen = False
                category_seen = False

                # print('-DOCSTART- ({} {})\n'.format(wiki_article['id'], wiki_article['title']))

                wiki_text_toks = list()
                wiki_text_toks_len = 0

                links_seen = 0
                kw_seen = 0
                inside_link = False
                entity = None
                inside_kw = False
                after_punct = True
                title_double_newline_seen = 0
                last_char = None

                reconstructed_wiki_text = ""
                current_snippet = ""

                if len(links_offsets) == 0:
                    continue

                if links_offsets[links_seen][0][0] == (len(reconstructed_wiki_text) + len(current_snippet)):
                    inside_link = True

                if len(keywords_found) > 0 and (
                    not inside_link
                    and (keywords_found[kw_seen][1] == (len(reconstructed_wiki_text) + len(current_snippet)))
                ):
                    inside_kw = True

                # if debug: print('inside_kw', inside_kw, 'inside_link', inside_link)

                for char_idx, char in enumerate(list(wiki_article["text"])):

                    if category_seen:
                        break

                    if char == "\n" and last_char != "." and not after_punct:
                        char = "."

                    if char == "." or last_char == ".":
                        after_punct = True
                    else:
                        after_punct = False

                    current_snippet = current_snippet + char

                    # if debug: print('wiki_text_toks', wiki_text_toks)

                    #
                    # check if *beginning* of annotated link or keyword link
                    #

                    if links_seen < len(links_offsets) and (
                        links_offsets[links_seen][0][0] == (len(reconstructed_wiki_text) + len(current_snippet))
                    ):

                        # clean up current snippet
                        if len(current_snippet) > 0:
                            current_snippet_tokenized = self.tokenizer.tokenize(current_snippet)
                            wiki_text_toks.extend(
                                zip(
                                    current_snippet_tokenized,
                                    ["O" for _ in current_snippet_tokenized],
                                    ["-" for _ in current_snippet_tokenized],
                                )
                            )
                            reconstructed_wiki_text += current_snippet
                            current_snippet = ""

                        # check if KB known entity
                        normalized_wiki_entity = normalize_wiki_entity(links_offsets[links_seen][1][1], replace_ws=True)
                        entity = self.redirects_en.get(normalized_wiki_entity, normalized_wiki_entity)
                        if entity in self.popular_entity_counter_dict:
                            inside_link = True
                        inside_kw = False

                    if kw_seen < len(keywords_found) and (
                        not inside_link
                        and (keywords_found[kw_seen][1] == (len(reconstructed_wiki_text) + len(current_snippet)))
                    ):

                        # clean up current snippet
                        if len(current_snippet) > 0:
                            current_snippet_tokenized = self.tokenizer.tokenize(current_snippet)
                            wiki_text_toks.extend(
                                zip(
                                    current_snippet_tokenized,
                                    ["O" for _ in current_snippet_tokenized],
                                    ["-" for _ in current_snippet_tokenized],
                                )
                            )
                            reconstructed_wiki_text += current_snippet
                            current_snippet = ""

                        inside_kw = True

                    #
                    # check if *end* of annotated link or keyword link
                    #

                    if links_seen < len(links_offsets) and (
                        links_offsets[links_seen][0][1] == (len(reconstructed_wiki_text) + len(current_snippet))
                    ):

                        # ignore if its Category link
                        if (
                            char_idx < len(wiki_article["text"]) - 1
                            and "Category:" in current_snippet + wiki_article["text"][char_idx + 1]
                        ):
                            category_seen = True
                            continue

                        # normalized_wiki_entity = normalize_wiki_entity(links_offsets[links_seen][1][1], replace_ws=True)
                        # entity = redirects_en.get(normalized_wiki_entity, normalized_wiki_entity)

                        if inside_link:
                            current_snippet_tokenized = self.tokenizer.tokenize(current_snippet)
                            wiki_text_toks.extend(
                                zip(
                                    current_snippet_tokenized,
                                    [entity for _ in current_snippet_tokenized],
                                    [current_snippet for _ in current_snippet_tokenized],
                                )
                            )
                            local_entities_found_counter[entity] += 1
                            local_mention_counter[current_snippet] += 1
                            reconstructed_wiki_text += current_snippet
                            current_snippet = ""

                        links_seen += 1
                        inside_link = False
                        entity = None

                    #
                    # check if *end* of keyword link and if any keyword matches have been seen.
                    #

                    if kw_seen < len(keywords_found) and (
                        keywords_found[kw_seen][2] == (len(reconstructed_wiki_text) + len(current_snippet))
                    ):

                        # ignore if its Category link
                        if (
                            char_idx < len(wiki_article["text"]) - 1
                            and "Category:" in current_snippet + wiki_article["text"][char_idx + 1]
                        ):
                            category_seen = True
                            continue

                        if inside_kw:
                            current_snippet_tokenized = self.tokenizer.tokenize(current_snippet)
                            if (
                                current_snippet in self.mention_entity_counter_popular_entities
                                and wiki_article_normalized_wiki_entity
                                in dict(self.mention_entity_counter_popular_entities[current_snippet])
                                and corefers_with_title_entity(current_snippet)
                            ):

                                local_mention_counter[current_snippet] += 1
                                wiki_text_toks.extend(
                                    zip(
                                        current_snippet_tokenized,
                                        [wiki_article_title_entity for _ in current_snippet_tokenized],
                                        [current_snippet for _ in current_snippet_tokenized],
                                    )
                                )
                            elif current_snippet in self.mention_entity_counter_popular_entities:
                                local_mention_counter[current_snippet] += 1
                                wiki_text_toks.extend(
                                    zip(
                                        current_snippet_tokenized,
                                        ["UNK" for _ in current_snippet_tokenized],
                                        [current_snippet for _ in current_snippet_tokenized],
                                    )
                                )
                            else:
                                wiki_text_toks.extend(
                                    zip(
                                        current_snippet_tokenized,
                                        ["O" for _ in current_snippet_tokenized],
                                        ["-" for _ in current_snippet_tokenized],
                                    )
                                )
                            reconstructed_wiki_text += current_snippet
                            current_snippet = ""

                            inside_kw = False
                            inside_link = False
                            entity = None

                        kw_seen += 1

                    last_char = char

                current_snippet_tokenized = self.tokenizer.tokenize(current_snippet)
                wiki_text_toks.extend(
                    zip(
                        current_snippet_tokenized,
                        ["O" for _ in current_snippet_tokenized],
                        ["-" for _ in current_snippet_tokenized],
                    )
                )
                reconstructed_wiki_text += current_snippet

                out_file_path = os.path.dirname(
                    f"data/versions/{self.opts.data_version_name}/wiki_training/raw/tmp/{file_name[len_prefix:]}"
                )
                if not os.path.exists(out_file_path):
                    os.makedirs(out_file_path, exist_ok=True)
                out_file_name = f"{out_file_path}/{wiki_article['id']}.tsv"
                pandas.DataFrame(wiki_text_toks).to_csv(out_file_name, sep="\t", header=None)
                out_file_names.append(out_file_name)
                local_entities_found_in_article.append((out_file_name, local_entities_found_counter))
        return out_file_names, local_mention_counter, local_entities_found_in_article
