import pickle
import sys
from collections import Counter, OrderedDict
from typing import Dict

import tqdm

sys.setrecursionlimit(10000)

from pipeline_job import PipelineJob


class PostProcessMentionEntityCounts(PipelineJob):
    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                f"data/versions/{opts.data_version_name}/indexes/mention_entity_counter.pickle",
                f"data/versions/{opts.data_version_name}/indexes/entity_counter.pickle",
                f"data/versions/{opts.data_version_name}/indexes/linked_mention_counter.pickle",
                f"data/versions/{opts.data_version_name}/indexes/found_conll_entities.pickle",
            ],
            provides=[
                f"data/versions/{opts.data_version_name}/indexes/mention_entity_counter_popular_entities.pickle",
                # was most_common_entity_counter_dict
                f"data/versions/{opts.data_version_name}/indexes/popular_entity_counter_dict.pickle",
                # was most_common_entity_counter_dict_to_id
                f"data/versions/{opts.data_version_name}/indexes/popular_entity_to_id_dict.pickle",
                # was all_mention__most_common_entity_id_probabilies
                f"data/versions/{opts.data_version_name}/indexes/mention_to_popular_entity_id_probabilies_dicts_dict.pickle",
        ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/entity_counter.pickle",
            "rb",
        ) as f:
            all_entity_counter = pickle.load(f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/mention_entity_counter.pickle",
            "rb",
        ) as f:
            all_mention_entity_counter = pickle.load(f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/found_conll_entities.pickle",
            "rb",
        ) as f:
            all_found_conll_entities = pickle.load(f)

        # Create the index over the most popular entities (configured by num_most_freq_entities)
        # Then create the mention entity index based on that

        # Create the index over the most popular entities
        popular_entity_counter_dict = dict(
            all_entity_counter.most_common()[: self.opts.num_most_freq_entities]
        )
        if self.opts.add_missing_conll_entities:
            # add entities required for the Aida-CoNLL benchmark dataset
            count = 0
            for ent in all_found_conll_entities:
                if ent in all_entity_counter:
                    popular_entity_counter_dict[ent] = all_entity_counter[ent]
                    count += 1
            self.log(f"Added {count} entities from the conll data back to the most popular entities vocabulary.")

        # Create the mention entity index based on that
        # TODO: filter rare entities for mentions / hacky heuristic to have cleaner data, can be improved
        mention_entity_counter_popular_entities = dict()
        for mention, entities in tqdm.tqdm(all_mention_entity_counter.items()):
            mention_entity_counter_popular_entities[mention] = Counter(
                {
                    k: v
                    for k, v in filter(
                        lambda t: t[0] in popular_entity_counter_dict and t[1] > 9,
                        entities.items(),
                    )
                }
            ).most_common()

        popular_entity_to_id_dict = OrderedDict(
            [
                (k, eid)
                for eid, (k, v) in enumerate(
                    Counter(popular_entity_counter_dict).most_common()
                )
            ]
        )

        mention_to_popular_entity_id_probabilies_dicts_dict = {
            m: {
                popular_entity_to_id_dict[ename]: count
                / sum([val for key, val in entities])
                for ename, count in entities
                if ename in popular_entity_to_id_dict
            }
            for m, entities in mention_entity_counter_popular_entities.items()
        }

        # was all_mention_entity_counter_most_common_entities
        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/mention_entity_counter_popular_entities.pickle",
            "wb",
        ) as f:
            pickle.dump(mention_entity_counter_popular_entities, f)

        # was most_common_entity_counter_dict
        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/popular_entity_counter_dict.pickle",
            "wb",
        ) as f:
            pickle.dump(popular_entity_counter_dict, f)

        # was most_common_entity_counter_dict_to_id
        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/popular_entity_to_id_dict.pickle",
            "wb",
        ) as f:
            pickle.dump(popular_entity_to_id_dict, f)

        # was all_mention__most_common_entity_id_probabilies
        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/mention_to_popular_entity_id_probabilies_dicts_dict.pickle",
            "wb",
        ) as f:
            pickle.dump(mention_to_popular_entity_id_probabilies_dicts_dict, f)
