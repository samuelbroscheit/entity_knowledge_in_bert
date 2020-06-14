import base64
import glob
import json
import io
import pickle
import sys
import multiprocessing
from collections import Counter
from typing import Dict

import pandas

from tqdm import tqdm
from misc import normalize_wiki_entity
from pipeline_job import PipelineJob


class CollectMentionEntityCounts(PipelineJob):
    """
    Collect mention entity counts from the Wikiextractor files.
    """
    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                "data/indexes/redirects_en.ttl.bz2.dict",
                f"data/versions/{opts.data_version_name}/wikiextractor_out/{opts.wiki_lang_version}/",
            ],
            provides=[
                f"data/versions/{opts.data_version_name}/indexes/mention_entity_counter.pickle",
                f"data/versions/{opts.data_version_name}/indexes/entity_counter.pickle",
                f"data/versions/{opts.data_version_name}/indexes/linked_mention_counter.pickle",
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        pandas.set_option("display.max_rows", 5000)
        sys.setrecursionlimit(10000)

        self.log("Load ./data/indexes/redirects_en.ttl.bz2.dict")
        with open(f"data/indexes/redirects_en.ttl.bz2.dict", "rb") as f:
            redirects_en = pickle.load(f)

        in_queue = multiprocessing.Queue()
        out_queue = multiprocessing.Queue()

        workers = list()

        list_dir_string = f"data/versions/{self.opts.data_version_name}/wikiextractor_out/{self.opts.wiki_lang_version}/{self.opts.wiki_lang_version}*pages-articles*/*/wiki_*"

        #
        # start the workers in individual processes
        #
        for id in range(self.opts.collect_mention_entities_num_workers):
            worker = Worker(in_queue, out_queue, redirects_en)
            worker.start()
            workers.append(worker)

        self.log("Fill queue")
        # fill the queue
        for file_nr, extracted_wiki_file in enumerate(tqdm(glob.glob(list_dir_string))):
            in_queue.put(extracted_wiki_file)
            self.debug("put {} in queue".format(extracted_wiki_file))

        all_linked_mention_counter = Counter()
        all_entity_counter = Counter()
        all_mention_entity_counter = dict()

        self.log("Collect the output")
        # collect the output
        for file_nr, extracted_wiki_file in enumerate(tqdm(glob.glob(list_dir_string))):
            (
                (
                    local_linked_mention_counter,
                    local_entity_counter,
                    local_mention_entity_counter,
                ),
                in_file_name,
            ) = out_queue.get()
            all_linked_mention_counter.update(local_linked_mention_counter)
            all_entity_counter.update(local_entity_counter)
            for k, v in local_mention_entity_counter.items():
                if k not in all_mention_entity_counter:
                    all_mention_entity_counter[k] = Counter()
                all_mention_entity_counter[k].update(v)

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

        with open(f"data/versions/{self.opts.data_version_name}/indexes/linked_mention_counter.pickle", "wb") as f:
            pickle.dump(all_linked_mention_counter, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/entity_counter.pickle", "wb") as f:
            pickle.dump(all_entity_counter, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/mention_entity_counter.pickle", "wb") as f:
            pickle.dump(all_mention_entity_counter, f)


class Worker(multiprocessing.Process):
    def __init__(self, in_queue, out_queue, redirects_en):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.redirects_en = redirects_en

    def run(self):
        # this loop will run until it receives None form the in_queue, if the queue is empty
        #  the loop will wait until it gets something
        for next_item in iter(self.in_queue.get, None):
            file_name = next_item
            self.out_queue.put((self.extract_data(next_item), file_name))

    def extract_data(self, file_name):

        local_entity_counter = Counter()
        local_linked_mention_counter = Counter()
        local_mention_entity_counter = Counter()

        with open(file_name) as f:

            for i, wiki_article in enumerate(f.readlines()):

                wiki_article = json.loads(wiki_article)

                start_offset_dict = dict()

                for ((start, end), (mention, wiki_page_name)) in pickle.loads(
                    base64.b64decode(wiki_article["internal_links"].encode("utf-8"))
                ).items():
                    start_offset_dict[start] = (end, (mention, wiki_page_name))
                    if mention.startswith("Category:"):
                        continue
                    normalized_wiki_entity = normalize_wiki_entity(
                        wiki_page_name, replace_ws=True
                    )
                    entity = self.redirects_en.get(
                        normalized_wiki_entity, normalized_wiki_entity
                    )
                    local_linked_mention_counter[mention] += 1
                    local_entity_counter[entity] += 1
                    if mention not in local_mention_entity_counter:
                        local_mention_entity_counter[mention] = Counter()
                    local_mention_entity_counter[mention][entity] += 1

        return (
            local_linked_mention_counter,
            local_entity_counter,
            local_mention_entity_counter,
        )
