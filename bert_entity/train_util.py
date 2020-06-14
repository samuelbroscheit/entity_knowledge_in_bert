import ast
import os

import configargparse as argparse
import torch.cuda
import yaml

from misc import argparse_bool_type

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", is_config_file=True, help="config file path")
parser.add_argument("--debug", type=argparse_bool_type, default=False)
parser.add_argument("--device", default=0)
parser.add_argument("--eval_device", default=None)
parser.add_argument("--dataset", default="EDLDataset")
parser.add_argument("--model", default="Net")
parser.add_argument("--data_version_name")
parser.add_argument("--wiki_lang_version")
parser.add_argument("--eval_on_test_only", type=argparse_bool_type, default=False)
parser.add_argument("--out_device", default=None)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--eval_batch_size", type=int, default=128)
parser.add_argument("--accumulate_batch_gradients", type=int, default=1)
parser.add_argument("--sparse", dest="sparse", type=argparse_bool_type)
parser.add_argument("--encoder_lr", type=float, default=5e-5)
parser.add_argument("--decoder_lr", type=float, default=1e-3)
parser.add_argument("--maskout_entity_prob", type=float, default=0)
parser.add_argument("--segm_decoder_lr", type=float, default=1e-3)
parser.add_argument("--encoder_weight_decay", type=float, default=0)
parser.add_argument("--decoder_weight_decay", type=float, default=0)
parser.add_argument("--segm_decoder_weight_decay", type=float, default=0)
parser.add_argument("--learn_segmentation", type=argparse_bool_type, default=False)
parser.add_argument("--label_size", type=int)
parser.add_argument("--vocab_size", type=int)
parser.add_argument("--entity_embedding_size", type=int, default=768)
parser.add_argument("--project", type=argparse_bool_type, default=False)
parser.add_argument("--n_epochs", type=int, default=1000)
parser.add_argument("--collect_most_popular_labels_steps", type=int, default=100)
parser.add_argument("--checkpoint_eval_steps", type=int, default=1000)
parser.add_argument("--checkpoint_save_steps", type=int, default=50000)
parser.add_argument("--finetuning", dest="finetuning", type=int, default=9999999999)
parser.add_argument("--top_rnns", dest="top_rnns", type=argparse_bool_type)
parser.add_argument("--logdir", type=str)
parser.add_argument("--train_loc_file", type=str, default="train.loc")
parser.add_argument("--valid_loc_file", type=str, default="valid.loc")
parser.add_argument("--test_loc_file", type=str, default="test.loc")
parser.add_argument("--resume_from_checkpoint", type=str)
parser.add_argument("--resume_reset_epoch", type=argparse_bool_type, default=False)
parser.add_argument("--resume_optimizer_from_checkpoint", type=argparse_bool_type, default=False)
parser.add_argument("--topk_neg_examples", type=int, default=3)
parser.add_argument("--dont_save_checkpoints", type=argparse_bool_type, default=False)
parser.add_argument("--data_workers", type=int, default=8)
parser.add_argument("--bert_dropout", type=float, default=None)
parser.add_argument("--encoder_lr_scheduler", type=str, default=None)
parser.add_argument("--encoder_lr_scheduler_config", default=None)
parser.add_argument("--decoder_lr_scheduler", type=str, default=None)
parser.add_argument("--decoder_lr_scheduler_config", default=None)
parser.add_argument("--segm_decoder_lr_scheduler", type=str, default=None)
parser.add_argument("--segm_decoder_lr_scheduler_config", default=None)
parser.add_argument("--eval_before_training", type=argparse_bool_type, default=False)
parser.add_argument("--data_path_conll", type=str,)
parser.add_argument("--train_data_dir", type=str, default="data")
parser.add_argument("--valid_data_dir", type=str, default="data")
parser.add_argument("--test_data_dir", type=str, default="data")
parser.add_argument("--exclude_parameter_names_regex", type=str)


def get_args():

    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(k, ":", v)
        if v == "None":
            args.__dict__[k] = None

    args.device = (
        int(args.device) if args.device is not None and args.device != "cpu" and torch.cuda.is_available() else "cpu"
    )
    if args.eval_device is not None:
        if args.eval_device != "cpu":
            args.eval_device = int(args.eval_device)
        else:
            args.eval_device = "cpu"
    else:
        args.eval_device = args.device
    if args.out_device is not None:
        if args.out_device != "cpu":
            args.out_device = int(args.out_device)
        else:
            args.out_device = "cpu"
    else:
        args.out_device = args.device

    if args.encoder_lr_scheduler_config:
        args.encoder_lr_scheduler_config = ast.literal_eval(args.encoder_lr_scheduler_config)
    if args.decoder_lr_scheduler_config:
        args.decoder_lr_scheduler_config = ast.literal_eval(args.decoder_lr_scheduler_config)
    if args.segm_decoder_lr_scheduler_config:
        args.segm_decoder_lr_scheduler_config = ast.literal_eval(args.segm_decoder_lr_scheduler_config)

    args.eval_batch_size = args.eval_batch_size if args.eval_batch_size else args.batch_size

    if not args.logdir:
        raise Exception("set args.logdir")

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    if not args.eval_on_test_only:
        config_fname = os.path.join(args.logdir, "config")
        with open(f"{config_fname}.yaml", "w") as f:
            f.writelines(
                [
                    "{}: {}\n".format(k, v)
                    for k, v in args.__dict__.items()
                    if isinstance(v, str) and len(v.strip()) > 0 or not isinstance(v, str) and v is not None
                ]
            )

    with open(f"data/versions/{args.data_version_name}/config.yaml") as f:
        dataset = yaml.load(f, Loader=yaml.UnsafeLoader)

    for k, v in dataset.items():
        if k != "debug":
            args.__setattr__(k, v)

    return args
