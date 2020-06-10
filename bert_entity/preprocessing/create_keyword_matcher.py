from collections import Counter, OrderedDict
from typing import Dict
import tqdm
import pickle
from flashtext import KeywordProcessor
import sys

sys.setrecursionlimit(10000)

from pipeline_job import PipelineJob


class CreateKeywordProcessor(PipelineJob):
    """
    Create a matcher to detect mentions that we found with Wikiextractor in free text.
    We use this later to add more annotations to the text. However, as we do not know
    the true entity, we'll associate labels for all entities from the with their
    p(e|m) prior.
    """
    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                f"data/versions/{opts.data_version_name}/indexes/mention_entity_counter_popular_entities.pickle",
            ],
            provides=[
                f"data/versions/{opts.data_version_name}/indexes/keyword_processor.pickle"
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/mention_entity_counter_popular_entities.pickle",
            "rb",
        ) as f:
            all_mention_entity_counter_most_popular_entities = pickle.load(f)


        keyword_processor = KeywordProcessor(case_sensitive=False)

        for (k, v_most_common) in tqdm.tqdm(
            list(all_mention_entity_counter_most_popular_entities.items())
        ):
            if (
                len(v_most_common) == 0
                or v_most_common is None
                or v_most_common[0] is None
                or v_most_common[0][0] is None
            ):
                continue
            if v_most_common[0][0].startswith("List"):
                continue
            if v_most_common[0][0].startswith("Category:"):
                continue
            if v_most_common[0][1] < 50:
                continue
            keyword_processor.add_keyword(k.replace("_", " "))
            keyword_processor.add_keyword(v_most_common[0][0].replace("_", " "))

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/keyword_processor.pickle",
            "wb"
        ) as f:
            pickle.dump(keyword_processor, f)
