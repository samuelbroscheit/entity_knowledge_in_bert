import bz2
import pickle
import re
from collections import defaultdict
from typing import Dict

import tqdm

from pipeline_job import PipelineJob


class CreateResolveToWikiNameDicts(PipelineJob):
    """
    Create a dictionary containing mapping Freebase Ids and Wikipedia pages ids
    to Wikipedia page names.
    Here we use the already extracted mapping from DBPedia that was created from
    a 2016 dump. The disambiguations are used to detect entity annotations in
    the AIDA-CONLL benchmark that have become incompatble for newer Wikipedia
    versions (Please note that in the expermiments for the paper a Wikipedia
    dump from 2017 was used. This dictionary might not adequate for the latest
    wiki dump).
    """
    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=["data/indexes/redirects_en.ttl.bz2.dict"],
            provides=[
                "data/indexes/freebase_links_en.ttl.bz2.dict",
                "data/downloads/freebase_links_en.ttl.bz2",
                "data/indexes/page_ids_en.ttl.bz2.dict",
                "data/downloads/page_ids_en.ttl.bz2",
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
        match_key=2,
        match_val=1,
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
                        ] = redirects[postproc_val(matcher_match.group(match_val))]
                    else:
                        a_to_b[
                            postproc_key(matcher_match.group(match_key))
                        ] = postproc_val(matcher_match.group(match_val))
        return a_to_b

    def _run(self):

        with open("data/indexes/redirects_en.ttl.bz2.dict", "rb") as f:
            redirects = pickle.load(f)

        fb_to_wikiname_dict = self._create_dict(
            redirects=redirects,
            url="http://downloads.dbpedia.org/2016-10/core-i18n/en/freebase_links_en.ttl.bz2",
            matcher_pattern="<http://dbpedia.org/resource/(.*)> <http://www.w3.org/2002/07/owl#sameAs> <http://rdf.freebase.com/ns/(.*)> .",
            postproc_key=lambda x: "/" + x.replace(".", "/"),
        )
        with open("data/indexes/freebase_links_en.ttl.bz2.dict", "wb") as f:
            pickle.dump(fb_to_wikiname_dict, f)

        page_id_to_wikiname_dict = self._create_dict(
            redirects=redirects,
            url="http://downloads.dbpedia.org/2016-10/core-i18n/en/page_ids_en.ttl.bz2",
            matcher_pattern='<http://dbpedia.org/resource/(.*)> <http://dbpedia.org/ontology/wikiPageID> "(.*)"\^\^<http://www.w3.org/2001/XMLSchema#integer> .',
        )
        with open("data/indexes/page_ids_en.ttl.bz2.dict", "wb") as f:
            pickle.dump(page_id_to_wikiname_dict, f)

