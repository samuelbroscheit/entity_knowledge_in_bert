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
                "",
                "--json",
                "--filter_disambig_pages",
                "--collect_links",
                "--processes",
                str(self.opts.wikiextractor_num_workers),
                input_file,
                "-o",
                f"data/versions/{self.opts.data_version_name}/wikiextractor_out/tmp/{os.path.basename(input_file)}",
            ]
            wiki_extractor_main()
        os.rename(
            f"data/versions/{self.opts.data_version_name}/wikiextractor_out/tmp/",
            f"data/versions/{self.opts.data_version_name}/wikiextractor_out/{self.opts.wiki_lang_version}/",
        )

        self.log("WikiExtractor finished")

