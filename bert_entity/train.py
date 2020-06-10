import copy
import logging
import os
import time

import torch.cuda
import torch.nn as nn

from metrics import Metrics
from data_loader_conll import CONLLEDLDataset
from data_loader_wiki import EDLDataset
from model import Net
from model_conll import ConllNet
from train_util import get_args
from vocab import Vocab


class Datasets:
    EDLDataset = EDLDataset
    CONLLEDLDataset = CONLLEDLDataset


class Models:
    Net = Net
    ConllNet = ConllNet


if __name__ == "__main__":

    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info(str(("Devices", args.device, args.eval_device, args.out_device)))

    # set up the model
    vocab = Vocab(args)
    model_class = getattr(Models, args.model)
    model = model_class(args=args, vocab_size=vocab.size())
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    if args.device != "cpu":
        torch.cuda.empty_cache()
        model.to(args.device, args.out_device)
    print(model)

    # set up the optimizers and the loss
    optimizers, lr_schedulers = model.get_optimizers(args, checkpoint=checkpoint)
    criterion = nn.BCEWithLogitsLoss()

    # set up the datasets and dataloaders
    if not args.eval_on_test_only:
        train_dataset = getattr(Datasets, args.dataset)(
            args, split="train", vocab=vocab, device=args.device, label_size=args.label_size
        )
        train_iter = train_dataset.get_data_iter(args=args, batch_size=args.batch_size, vocab=vocab, train=True)
        eval_dataset = getattr(Datasets, args.dataset)(args, split="valid", vocab=vocab, device=args.eval_device)
        eval_iter = eval_dataset.get_data_iter(args=args, batch_size=args.eval_batch_size, vocab=vocab, train=False)
    else:
        eval_dataset = getattr(Datasets, args.dataset)(args, split="test", vocab=vocab, device=args.eval_device)
        eval_iter = eval_dataset.get_data_iter(args=args, batch_size=args.eval_batch_size, vocab=vocab, train=False)

    start_epoch = 1
    if checkpoint and not args.resume_reset_epoch:
        start_epoch = checkpoint["epoch"]

    metrics = Metrics()

    if args.eval_before_training or args.eval_on_test_only:
        cloned_args = copy.deepcopy(args)
        cloned_args.dont_save_checkpoints = True
        metrics = model_class.evaluate(
            cloned_args,
            model,
            eval_iter,
            optimizers=optimizers,
            step=0,
            epoch=0,
            save_checkpoint=False,
            save_csv=args.eval_on_test_only,
            vocab=vocab,
            metrics=metrics,
        )

    if not args.eval_on_test_only:
        for epoch in range(start_epoch, args.n_epochs + 1):

            start = time.time()

            model.finetuning = epoch >= args.finetuning if args.finetuning >= 0 else False

            metrics = model_class.train_one_epoch(
                args=args,
                model=model,
                train_iter=train_iter,
                optimizers=optimizers,
                criterion=criterion,
                vocab=vocab,
                eval_iter=eval_iter,
                epoch=epoch,
                metrics=metrics,
            )

            logging.info(f"Evaluate in epoch {epoch}")
            metrics = model_class.evaluate(
                args, model, eval_iter, optimizers=optimizers, step=0, epoch=epoch, vocab=vocab, metrics=metrics,
            )

            logging.info(f"{time.time() - start} per epoch")

            if lr_schedulers:
                for lr_scheduler in lr_schedulers:
                    lr_scheduler.step(metrics.get_model_selection_metric())
