import os

from preprocessing.create_integerized_aida_conll_training import CreateIntegerizedCONLLTrainingData
from preprocessing.preprocess_aida_conll_data import CreateAIDACONLL
from preprocessing.create_integerized_wiki_training import CreateIntegerizedWikiTrainingData
from preprocessing.create_keyword_matcher import CreateKeywordProcessor
from preprocessing.create_disambiguation_dict import CreateDisambiguationDict
from preprocessing.create_resolve_to_wiki_dicts import CreateResolveToWikiNameDicts
from preprocessing.create_wiki_training_data import CreateWikiTrainingData
from preprocessing.postprocess_mention_entity_counts import PostProcessMentionEntityCounts
from pipeline_job import PipelineJob
from preprocessing.collect_mention_entity_counts import CollectMentionEntityCounts
from preprocessing.create_redirects import CreateRedirects
from preprocessing.download_data import DownloadWikiDump
from preprocessing.wikiextractor import Wikiextractor
from misc import argparse_bool_type
import configargparse as argparse
import logging


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", is_config_file=True, help="config file path")
parser.add_argument("--debug", type=argparse_bool_type, default=False)
parser.add_argument("--wiki_lang_version", type=str, help="wiki language version", default="enwiki")
parser.add_argument("--data_version_name", type=str, help="data identifier/version")
parser.add_argument("--download_data_only_dummy", type=argparse_bool_type, help="only download one wiki file")
parser.add_argument("--download_2017_enwiki", type=argparse_bool_type, help="download the enwiki 2017 dump to reproduce the experiments for the CONLL 2019 paper", default=True)
parser.add_argument("--num_most_freq_entities", type=int, help="")
parser.add_argument("--add_missing_conll_entities", type=argparse_bool_type, help="")
parser.add_argument("--uncased", type=argparse_bool_type, default=True)

parser.add_argument("--collect_mention_entities_num_workers", type=int, default="10")

parser.add_argument("--wikiextractor_num_workers", type=int, help="")

parser.add_argument("--create_training_data_num_workers", type=int, default="10")
parser.add_argument("--create_training_data_num_entities_in_necessary_articles", type=int, help="")
parser.add_argument("--create_training_data_discount_nil_strategy", type=str, help="the discount strategy either 'hacky' or 'prop'", default="prop")

parser.add_argument("--create_integerized_training_num_workers", type=int, default="10")
parser.add_argument("--create_integerized_training_loc_file_name", type=str, default="data.loc")
parser.add_argument("--create_integerized_training_instance_text_length", type=int, default="254")
parser.add_argument("--create_integerized_training_instance_text_overlap", type=int, default="20")
parser.add_argument("--create_integerized_training_max_entity_per_shard_count", type=int, default="10")
parser.add_argument("--create_integerized_training_valid_size", type=int, default="1000")
parser.add_argument("--create_integerized_training_test_size", type=int, default="1000")

args = parser.parse_args()

if args.download_2017_enwiki:
    if len(args.wiki_lang_version) > 0 and args.wiki_lang_version != 'enwiki':
        raise Exception(f"The configuration was set to 'download_2017_enwiki=True' but wiki_lang_version was set to {args.wiki_lang_version}.")

if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

for k, v in args.__dict__.items():
    logging.info(f"{k}: {v}")
    if v == "None":
        args.__dict__[k] = None

os.makedirs(f"data/versions/{args.data_version_name}/", exist_ok=True)

with open(f"data/versions/{args.data_version_name}/config.yaml", "w") as f:
    f.writelines(["{}: {}\n".format(k, v) for k, v in args.__dict__.items()])

PipelineJob.run_jobs([
    CreateRedirects,
    CreateResolveToWikiNameDicts,
    CreateDisambiguationDict,
    DownloadWikiDump,
    Wikiextractor,
    CollectMentionEntityCounts,
    PostProcessMentionEntityCounts,
    CreateAIDACONLL,
    CreateKeywordProcessor,
    CreateWikiTrainingData,
    CreateIntegerizedWikiTrainingData,
    CreateIntegerizedCONLLTrainingData,
], args)
