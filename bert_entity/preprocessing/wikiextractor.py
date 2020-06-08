import base64
import glob
import io
import json
import multiprocessing
import os
import pickle
import sys
from collections import Counter
from typing import Dict

from WikiExtractor import main as wiki_extractor_main
from misc import normalize_wiki_entity
from pipeline_job import PipelineJob


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

        with io.open(file_name) as f:

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


class Wikiextractor(PipelineJob):
    """
    Run Wikiextractor on the Wikipedia dump and extract all the mentions from it.
    """
    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                f"data/versions/{opts.data_version_name}/downloads/{opts.wiki_lang_version}/",
            ],
            provides=[
                f"data/versions/{opts.data_version_name}/wikiextractor_out/{opts.wiki_lang_version}/",
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        self.log("Run WikiExtractor")

        # python wikiextractor-wikimentions/WikiExtractor.py --json --filter_disambig_pages --processes $WIKI_EXTRACTOR_NR_PROCESSES --collect_links $DOWNLOADS_DIR/$WIKI_RAW/$WIKI_FILE -o $WIKI_EXTRACTOR_OUTDIR/$WIKI_FILE

        for input_file in glob.glob(
            f"data/versions/{self.opts.data_version_name}/downloads/{self.opts.wiki_lang_version}/*"
        ):
            self.log(input_file)
            sys.argv = [
                "--json",
                "--filter_disambig_pages",
                "--collect_links",
                "--processes",
                str(self.opts.wikiextractor_num_workers),
                input_file,
                "-o",
                f"data/versions/{self.opts.data_version_name}/extracted_mentions/tmp/{os.path.basename(input_file)}",
            ]
            wiki_extractor_main()
        os.rename(
            f"data/versions/{self.opts.data_version_name}/wikiextractor_out/tmp/",
            f"data/versions/{self.opts.data_version_name}/wikiextractor_out/{self.opts.wiki_lang_version}/",
        )

        self.log("WikiExtractor finished")
