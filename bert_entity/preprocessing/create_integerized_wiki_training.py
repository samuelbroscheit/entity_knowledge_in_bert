import multiprocessing
import os
import pickle
import random
from collections import Counter
from itertools import cycle
from typing import Dict

from tqdm import tqdm

from misc import unescape, create_chunks, create_overlapping_chunks
from pipeline_job import PipelineJob
from vocab import Vocab


class CreateIntegerizedWikiTrainingData(PipelineJob):
    """
    Create overlapping chunks of the Wikipedia articles. Outputs are stored as
    Python lists with integer ids. Configured by "create_integerized_training_instance_text_length"
    and "create_integerized_training_instance_text_overlap".

    Each worker creates his own shard, i.e., the number of shards is determined by
    "create_integerized_training_num_workers".

    Only save a training instance (a chunk of a Wikipedia article) if at least one entity in that
    chunk has not been seen more than "create_integerized_training_max_entity_per_shard_count" times.
    This downsamples highly frequent entities. Has to be set in relation to "create_integerized_training_num_workers"
    and "num_most_freq_entities". For the CONLL 2019 paper experiments the setting was

    create_integerized_training_max_entity_per_shard_count = 10
    create_integerized_training_num_workers = 40
    num_most_freq_entities = 500000
    """
    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                "data/indexes/redirects_en.ttl.bz2.dict",
                f"data/versions/{opts.data_version_name}/indexes/keyword_processor.pickle",
                f"data/versions/{opts.data_version_name}/indexes/popular_entity_counter_dict.pickle",
                f"data/versions/{opts.data_version_name}/indexes/mention_entity_counter_popular_entities.pickle",
                f"data/versions/{opts.data_version_name}/indexes/popular_entity_to_id_dict.pickle",
                f"data/versions/{opts.data_version_name}/indexes/mention_entity_discounted_probs.pickle",
                f"data/versions/{opts.data_version_name}/indexes/necessary_articles.pickle",
            ],
            provides=[f"data/versions/{opts.data_version_name}/wiki_training/integerized/{opts.wiki_lang_version}/"],
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
            mention_entity_counter_popular_entities = pickle.load(f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/popular_entity_to_id_dict.pickle", "rb") as f:
            popular_entity_to_id_dict = pickle.load(f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/mention_entity_discounted_probs.pickle", "rb"
        ) as f:
            me_pop_p = pickle.load(f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/necessary_articles.pickle", "rb") as f:
            necessary_articles = pickle.load(f)

        out_dir = f"data/versions/{self.opts.data_version_name}/wiki_training/integerized/tmp/"

        vocab = Vocab()
        vocab.load(self.opts, popular_entity_to_id_dict=popular_entity_to_id_dict)

        in_queue = multiprocessing.Queue()
        out_queue = multiprocessing.Queue()

        workers = list()

        #
        # start the workers in individual processes
        #

        shards = cycle(range(self.opts.create_integerized_training_num_workers))

        for id in range(self.opts.create_integerized_training_num_workers):
            worker = Worker(
                in_queue,
                out_queue,
                shard=next(shards),
                opts=self.opts,
                out_dir=out_dir,
                redirects_en=redirects_en,
                keyword_processor=keyword_processor,
                popular_entity_counter_dict=most_popular_entity_counter_dict,
                mention_entity_counter_popular_entities=mention_entity_counter_popular_entities,
                me_pop_p=me_pop_p,
                vocab=vocab,
            )
            worker.start()
            workers.append(worker)

        self.log("Fill queue")

        submitted_jobs = 0

        for file_nr, extracted_wiki_file in enumerate(tqdm(necessary_articles)):
            submitted_jobs += 1
            in_queue.put((extracted_wiki_file))

        self.log("Collect the output")

        joined_data_loc_list = list()

        for _ in tqdm(range(submitted_jobs)):
            (local_joined_data_loc_list), in_file_name = out_queue.get()
            if (
                local_joined_data_loc_list is not None
                and len(local_joined_data_loc_list) > 0
                and local_joined_data_loc_list[0] is not None
            ):
                joined_data_loc_list.extend(local_joined_data_loc_list)

        with open(f"{out_dir}/data.loc", "w") as f_loc:
            f_loc.writelines(joined_data_loc_list)

        random.shuffle(joined_data_loc_list)

        with open(f"{out_dir}/train.loc", "w") as f_loc:
            f_loc.writelines(
                joined_data_loc_list[
                    : -(
                        self.opts.create_integerized_training_valid_size
                        + self.opts.create_integerized_training_test_size
                    )
                ]
            )

        with open(f"{out_dir}/valid.loc", "w") as f_loc:
            f_loc.writelines(
                joined_data_loc_list[
                    -(
                        self.opts.create_integerized_training_valid_size
                        + self.opts.create_integerized_training_test_size
                    ) : -self.opts.create_integerized_training_test_size
                ]
            )

        with open(f"{out_dir}/test.loc", "w") as f_loc:
            f_loc.writelines(joined_data_loc_list[-self.opts.create_integerized_training_test_size :])

        # put the None into the queue so the loop in the run() function of the worker stops
        for worker in workers:
            in_queue.put(None)
            out_queue.put(None)

        # terminate the process
        for worker in workers:
            worker.join()

        os.rename(
            f"data/versions/{self.opts.data_version_name}/wiki_training/integerized/tmp/",
            f"data/versions/{self.opts.data_version_name}/wiki_training/integerized/{self.opts.wiki_lang_version}/",
        )


class Worker(multiprocessing.Process):
    def __init__(
        self,
        in_queue,
        out_queue,
        shard,
        opts,
        out_dir,
        redirects_en,
        keyword_processor,
        popular_entity_counter_dict,
        mention_entity_counter_popular_entities,
        me_pop_p,
        vocab,
    ):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.shard = shard
        self.redirects_en = redirects_en
        self.keyword_processor = keyword_processor
        self.popular_entity_counter_dict = popular_entity_counter_dict
        self.mention_entity_counter_popular_entities = mention_entity_counter_popular_entities
        self.me_pop_p = me_pop_p
        self.vocab = vocab
        self.opts = opts
        out_file = f"{out_dir}/{self.shard}.dat"
        os.makedirs(out_dir, exist_ok=True)
        self.pickle_file = open(out_file, "wb")
        self.entity_counts = Counter()

    def run(self):
        # this loop will run until it receives None form the in_queue, if the queue is empty
        for next_item in iter(self.in_queue.get, None):
            file_name = next_item
            self.out_queue.put((self.extract_data(next_item), file_name))
        self.pickle_file.close()

    def extract_data(self, file_name):

        instance_text_length = self.opts.create_integerized_training_instance_text_length
        instance_text_overlap = self.opts.create_integerized_training_instance_text_overlap
        max_entity_per_shard_count = self.opts.create_integerized_training_max_entity_per_shard_count

        local_joined_data_loc_list = list()

        if os.path.getsize(file_name) == 0:
            return None, None, None, None

        def map_func(line):
            items = line.strip().split("\t")
            if len(items) != 4:
                return "[UNK]", "O", "-"
            else:
                _, tok, ent, ment = items
                tok = unescape(tok)
                ent = unescape(ent)
                return tok if tok else "[UNK]", ent if ent else "O", ment if ment else "-"

        with open(file_name) as f:
            toks, ents, ments = zip(*map(map_func, f.readlines()))

        token_ids = list()
        for chunk in create_chunks(toks, 512):
            token_ids.extend(self.vocab.tokenizer.convert_tokens_to_ids(chunk))

        mention_entity_ids, mention_entity_probs, mention_probs, is_entity = list(), list(), list(), list()

        # if df.isnull().values.any():
        #     print(file_name, '\n', df)
        #     return None, None, None, None

        for i, (entity, mention) in enumerate(zip(ents, ments)):
            if (
                entity == "O"
                or entity == "UNK"
                and (
                    mention not in self.mention_entity_counter_popular_entities
                    or len(self.mention_entity_counter_popular_entities[mention]) == 0
                )
            ):
                mention_entity_ids.append(([self.vocab.OUTSIDE_ID],))  #
                mention_entity_probs.append(([1.0],))
                mention_probs.append((1.0,))
                is_entity.append(1000000000)
            elif entity == "UNK" and mention in self.me_pop_p:
                this_mention_entity_ids, this_mention_entity_probs = zip(*list(self.me_pop_p[mention]))
                mention_entity_ids.append((list(map(self.vocab.tag2idx.__getitem__, this_mention_entity_ids)),))
                mention_entity_probs.append((this_mention_entity_probs,))
                mention_probs.append((1.0, 1.0,))
                is_entity.append(1000000000)
            else:
                mention_entity_ids.append(([self.vocab.tag2idx[entity]],))
                mention_entity_probs.append(([1.0],))
                mention_probs.append((1.0,))
                is_entity.append(self.vocab.tag2idx[entity])

        for (
            token_ids_chunk,
            mention_entity_ids_chunk,
            mention_entity_probs_chunk,
            mention_probs_chunk,
            is_entity_chunk,
        ) in zip(
            create_overlapping_chunks(token_ids, instance_text_length, instance_text_overlap),
            create_overlapping_chunks(mention_entity_ids, instance_text_length, instance_text_overlap),
            create_overlapping_chunks(mention_entity_probs, instance_text_length, instance_text_overlap),
            create_overlapping_chunks(mention_probs, instance_text_length, instance_text_overlap),
            create_overlapping_chunks(is_entity, instance_text_length, instance_text_overlap),
        ):
            # Only save a training instance (a chunk of a Wikipedia article) if at least one entity in that
            # chunk has not been seen more than "max_entity_per_shard_count" times. This downsamples highly
            # frequent entities.
            if (
                sum(map(lambda i: 1 if self.entity_counts[i] < (max_entity_per_shard_count + 1) else 0, is_entity_chunk,))
                > 0
            ):
                self.entity_counts.update(is_entity_chunk)
                local_joined_data_loc_list.append(
                    str("{}\t{}\n".format(self.shard, self.pickle_file.tell()))
                )  # remember row byte offset
                pickle.dump(
                    (token_ids_chunk, mention_entity_ids_chunk, mention_entity_probs_chunk, mention_probs_chunk),
                    self.pickle_file,
                )  # write new row
        self.pickle_file.flush()
        return local_joined_data_loc_list


