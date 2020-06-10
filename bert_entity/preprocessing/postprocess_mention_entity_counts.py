import pickle
import sys
from collections import Counter, OrderedDict
from typing import Dict

import tqdm

sys.setrecursionlimit(10000)

from pipeline_job import PipelineJob


class PostProcessMentionEntityCounts(PipelineJob):
    """
    Create entity indexes that will later be used in the creation of the Wikipedia
    training data.
    First, based on the configuration key "num_most_freq_entities" the top k most
    popular entities are selected. Based on those, other mappings are created to only
    contain counts and priors concerning the top k popular entities. Later the top k
    popular entities will also restrict the training  data to only contain instances
    that contain popular entities.
    Also, if "add_missing_conll_entities" is set, the entity ids that are missing
    in the top k popular entities we'll add the entities that are missing in the AIDA-CONLL
    benchmark to ensure that the evaluation measures are comparable to prior work.
    """
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
                f"data/versions/{opts.data_version_name}/indexes/popular_entity_counter_dict.pickle",
                f"data/versions/{opts.data_version_name}/indexes/popular_entity_to_id_dict.pickle",
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

        # Create a mapping from entities to ids
        popular_entity_to_id_dict = OrderedDict(
            [
                (k, eid)
                for eid, (k, v) in enumerate(
                    Counter(popular_entity_counter_dict).most_common()
                )
            ]
        )

        # Create a dictionary for the prior probablities p(e|m) of mentions to
        # ids of popular entities
        mention_to_popular_entity_id_probabilies_dicts_dict = {
            m: {
                popular_entity_to_id_dict[ename]: count
                / sum([val for key, val in entities])
                for ename, count in entities
                if ename in popular_entity_to_id_dict
            }
            for m, entities in mention_entity_counter_popular_entities.items()
        }

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/mention_entity_counter_popular_entities.pickle",
            "wb",
        ) as f:
            pickle.dump(mention_entity_counter_popular_entities, f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/popular_entity_counter_dict.pickle",
            "wb",
        ) as f:
            pickle.dump(popular_entity_counter_dict, f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/popular_entity_to_id_dict.pickle",
            "wb",
        ) as f:
            pickle.dump(popular_entity_to_id_dict, f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/mention_to_popular_entity_id_probabilies_dicts_dict.pickle",
            "wb",
        ) as f:
            pickle.dump(mention_to_popular_entity_id_probabilies_dicts_dict, f)
