import bz2
import pickle
import re
from collections import defaultdict
from typing import Dict

import tqdm

from pipeline_job import PipelineJob


class CreateDisambiguationDict(PipelineJob):
    """
    Create a dictionary containing disambiguations for Wikipedia page names.
    Here we use the already extracted mapping from DBPedia that was created from
    a 2016 dump. The disambiguations are used to detect entity annotations in
    the AIDA-CONLL benchmark that have become incompatble for newer Wikipedia
    versions (I was using a Wikipedia dump from 2017. This dictionary might not
    be that fitting for the current wiki dump).
    """
    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=["data/indexes/redirects_en.ttl.bz2.dict"],
            provides=[
                "data/indexes/disambiguations_en.ttl.bz2.dict",
                "data/downloads/disambiguations_en.ttl.bz2",
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _create_dict(
        self,
        redirects,
        url,
        matcher_pattern,
        postproc_key=lambda x: x,
        postproc_val=lambda x: x,
        match_key=1,
        match_val=2,
    ):
        downloaded = self._download(url, "data/downloads/",)
        matcher = re.compile(matcher_pattern)

        a_to_b = defaultdict(list)
        with bz2.BZ2File(downloaded, "rb") as file:
            for line in tqdm.tqdm(file):
                line_decoded = line.decode().strip()
                matcher_match = matcher.match(line_decoded)
                if matcher_match:
                    if matcher_match.group(match_val) in redirects:
                        a_to_b[
                            postproc_key(matcher_match.group(match_key))
                        ].append(redirects[postproc_val(matcher_match.group(match_val))])
                    else:
                        a_to_b[
                            postproc_key(matcher_match.group(match_key))
                        ].append(postproc_val(matcher_match.group(match_val)))
        return a_to_b

    def _run(self):

        with open("data/indexes/redirects_en.ttl.bz2.dict", "rb") as f:
            redirects = pickle.load(f)

        fb_to_wikiname_dict = self._create_dict(
            redirects=redirects,
            url="http://downloads.dbpedia.org/2016-10/core-i18n/en/disambiguations_en.ttl.bz2",
            matcher_pattern="<http://dbpedia.org/resource/(.*)> <http://dbpedia.org/ontology/wikiPageDisambiguates> <http://dbpedia.org/resource/(.*)> .",
        )
        with open("data/indexes/disambiguations_en.ttl.bz2.dict", "wb") as f:
            pickle.dump(fb_to_wikiname_dict, f)

