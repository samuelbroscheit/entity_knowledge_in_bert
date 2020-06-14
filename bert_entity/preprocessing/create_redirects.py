import bz2
import io
import os
import pickle
import re
import urllib.request
from typing import Dict

import tqdm

from pipeline_job import PipelineJob


class CreateRedirects(PipelineJob):
    """
    Create a dictionary containing redirects for Wikipedia page names. Here we use
    the already extracted mapping from DBPedia that was created from a 2016 dump.
    The redirects are used for the Wikipedia mention extractions as well as for
    the AIDA-CONLL benchmark.
    """
    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[],
            provides=[
                "data/indexes/redirects_en.ttl.bz2.dict",
                "data/downloads/redirects_en.ttl.bz2",
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):
        self._download(
            "http://downloads.dbpedia.org/2016-10/core-i18n/en/redirects_en.ttl.bz2",
            "data/downloads/",
        )

        redirects = dict()
        redirects_first_sweep = dict()

        redirect_matcher = re.compile(
            "<http://dbpedia.org/resource/(.*)> <http://dbpedia.org/ontology/wikiPageRedirects> <http://dbpedia.org/resource/(.*)> ."
        )

        with bz2.BZ2File("data/downloads/redirects_en.ttl.bz2", "rb") as file:
            for line in tqdm.tqdm(file.readlines()):
                line_decoded = line.decode().strip()
                redirect_matcher_match = redirect_matcher.match(line_decoded)
                if redirect_matcher_match:
                    redirects[
                        redirect_matcher_match.group(1)
                    ] = redirect_matcher_match.group(2)
                    redirects_first_sweep[
                        redirect_matcher_match.group(1)
                    ] = redirect_matcher_match.group(2)
                # else:
                #     print(line_decoded)

        with bz2.BZ2File("data/downloads/redirects_en.ttl.bz2", "rb") as file:
            for line in tqdm.tqdm(file.readlines()):
                line_decoded = line.decode().strip()
                redirect_matcher_match = redirect_matcher.match(line_decoded)
                if redirect_matcher_match:
                    if redirect_matcher_match.group(2) in redirects:
                        redirects[redirect_matcher_match.group(1)] = redirects[
                            redirect_matcher_match.group(2)
                        ]

        with io.open("data/indexes/redirects_en.ttl.bz2.dict", "wb") as f:
            pickle.dump(redirects, f)
