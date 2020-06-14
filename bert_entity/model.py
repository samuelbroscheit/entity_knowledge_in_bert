import logging
import os
from itertools import chain

import numpy
import torch
import torch.nn as nn
import tqdm
from torch import optim
from tqdm import trange

from metrics import Metrics
from data_loader_wiki import EDLDataset_collate_func
from misc import running_mean, get_topk_ids_aggregated_from_seq_prediction, DummyOptimizer, LRSchedulers
from pytorch_pretrained_bert import BertModel


class Net(nn.Module):
    def __init__(
        self, args, vocab_size=None,
    ):
        super().__init__()
        if args.uncased:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert = BertModel.from_pretrained("bert-base-cased")

        self.top_rnns = args.top_rnns
        if args.top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768 // 2, batch_first=True)
        self.fc = None
        if args.project:
            self.fc = nn.Linear(768, args.entity_embedding_size)
        self.out = nn.Embedding(num_embeddings=vocab_size, embedding_dim=args.entity_embedding_size, sparse=args.sparse)
        # torch.nn.init.normal_(self.out, std=0.1)

        self.device = args.device
        self.out_device = args.out_device
        self.finetuning = args.finetuning == 0
        self.vocab_size = vocab_size

    def to(self, device, out_device):
        self.bert.to(device)
        if self.fc:
            self.fc.to(device)
        self.out.to(out_device)
        self.device = device
        self.out_device = out_device

    def forward(self, x, y=None, probs=None, enc=None):
        """
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        """
        if y is not None:
            y = y.to(self.out_device)
        if probs is not None:
            probs = probs.to(self.out_device)

            # fake_y = torch.Tensor(range(10)).long().to(self.device)

        if enc is None:
            x = x.to(self.device)
            if self.training:
                if self.finetuning:
                    # print("->bert.train()")
                    self.bert.train()
                    encoded_layers, _ = self.bert(x)
                    enc = encoded_layers[-1]
                else:
                    self.bert.eval()
                    with torch.no_grad():
                        encoded_layers, _ = self.bert(x)
                        enc = encoded_layers[-1]
            else:
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]

            if self.top_rnns:
                enc, _ = self.rnn(enc)

            if self.fc:
                enc = self.fc(enc)

            enc = enc.to(self.out_device)

        if y is not None:
            out = self.out(y)
            logits = enc.matmul(out.transpose(0, 1))
            y_hat = logits.argmax(-1)
            return logits, y, y_hat, probs, out, enc
        else:
            with torch.no_grad():
                out = self.out.weight
                logits = enc.matmul(out.transpose(0, 1))
                y_hat = logits.argmax(-1)
                return logits, None, y_hat, None, None, enc

    @staticmethod
    def train_one_epoch(
        args,
        model,
        train_iter,
        optimizers,
        criterion,
        eval_iter,
        vocab,
        epoch,
        metrics=Metrics(),
        loss_aggr=None,
    ):
        labels_with_high_model_score = None

        with trange(len(train_iter)) as t:
            for iter, batch in enumerate(train_iter):

                model.to(
                    args.device, args.out_device,
                )
                model.train()

                batch_token_ids, label_ids, label_probs, eval_mask, _, _, orig_batch, loaded_batch = batch

                enc = None

                if (
                    args.collect_most_popular_labels_steps is not None
                    and args.collect_most_popular_labels_steps > 0
                    and iter > 0
                    and iter % args.collect_most_popular_labels_steps == 0
                ):
                    model.to(args.device, args.eval_device)
                    with torch.no_grad():
                        logits_, _, _, _, _, enc = model(
                            batch_token_ids, None, None,
                        )  # logits: (N, T, VOCAB), y: (N, T)
                        labels_with_high_model_score = get_topk_ids_aggregated_from_seq_prediction(
                            logits_, topk_from_batch=args.label_size, topk_per_token=args.topk_neg_examples
                        )
                        batch_token_ids, label_ids, label_probs, eval_mask, _, _, _, _ = EDLDataset_collate_func(
                            args=args,
                            labels_with_high_model_score=labels_with_high_model_score,
                            batch=orig_batch,
                            return_labels=True,
                            vocab=vocab,
                            is_training=False,
                            loaded_batch=loaded_batch,
                        )

                # if args.label_size is not None:
                logits, y, y_hat, label_probs, sparse_params, _ = model(
                    batch_token_ids, label_ids, label_probs, enc=enc
                )  # logits: (N, T, VOCAB), y: (N, T)
                logits = logits.view(-1)  # (N*T, VOCAB)
                label_probs = label_probs.view(-1)  # (N*T,)

                loss = criterion(logits, label_probs)

                loss.backward()

                if (iter + 1) % args.accumulate_batch_gradients == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()

                if iter == 0:
                    logging.debug(f"Sanity check")
                    logging.debug("x:", batch_token_ids.cpu().numpy()[0])
                    logging.debug("tokens:", vocab.tokenizer.convert_ids_to_tokens(batch_token_ids.cpu().numpy()[0]))
                    logging.debug("y:", label_probs.cpu().numpy()[0])

                loss_aggr = running_mean(loss.detach().item(), loss_aggr)

                if iter > 0 and iter % args.checkpoint_eval_steps == 0:
                    metrics = Net.evaluate(
                        args=args,
                        model=model,
                        iterator=eval_iter,
                        optimizers=optimizers,
                        step=iter,
                        epoch=epoch,
                        save_checkpoint=iter % args.checkpoint_save_steps == 0,
                        sampled_evaluation=False,
                        metrics=metrics,
                        vocab=vocab,
                    )

                t.set_postfix(
                    loss=loss_aggr,
                    nr_labels=len(label_ids),
                    aggr_labels=len(labels_with_high_model_score) if labels_with_high_model_score else 0,
                    last_eval=metrics.report(filter={"f1", "num_proposed", "epoch", "step"}),
                )
                t.update()

        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        return metrics

    @staticmethod
    def evaluate(
        args,
        model,
        iterator,
        vocab,
        optimizers,
        step=0,
        epoch=0,
        save_checkpoint=True,
        save_predictions=True,
        save_csv=True,
        sampled_evaluation=False,
        metrics=Metrics(),
    ):

        print()
        logging.info(f"Start evaluation on split {'test' if args.eval_on_test_only else 'valid'}")

        model.eval()
        model.to(args.device, args.eval_device)

        all_words, all_tags, all_y, all_y_hat, all_predicted, all_token_ids = [], [], [], [], [], []
        with torch.no_grad():
            for iter, batch in enumerate(tqdm.tqdm(iterator)):
                (
                    batch_token_ids,
                    label_ids,
                    label_probs,
                    eval_mask,
                    label_id_to_entity_id_dict,
                    batch_entity_ids,
                    orig_batch,
                    _,
                ) = batch

                logits, y, y_hat, probs, _, _ = model(batch_token_ids, None, None)  # logits: (N, T, VOCAB), y: (N, T)

                tags = list()
                predtags = list()
                y_resolved_list = list()
                y_hat_resolved_list = list()
                token_list = list()

                chunk_len = args.create_integerized_training_instance_text_length
                chunk_overlap = args.create_integerized_training_instance_text_overlap

                for batch_id, seq in enumerate(label_probs.max(-1)[1]):
                    for tok_id, label_id in enumerate(seq[chunk_overlap : -chunk_overlap]):
                        y_resolved = (
                            vocab.PAD_ID
                            if eval_mask[batch_id][tok_id + chunk_overlap] == 0
                            else label_ids[label_id].item()
                        )
                        y_resolved_list.append(y_resolved)
                        tags.append(vocab.idx2tag[y_resolved])
                        if sampled_evaluation:
                            y_hat_resolved = (
                                vocab.PAD_ID
                                if eval_mask[batch_id][tok_id + chunk_overlap] == 0
                                else label_ids[y_hat[batch_id][tok_id + chunk_overlap]].item()
                            )
                        else:
                            y_hat_resolved = y_hat[batch_id][tok_id + chunk_overlap].item()
                        y_hat_resolved_list.append(y_hat_resolved)
                        predtags.append(vocab.idx2tag[y_hat_resolved])
                        token_list.append(batch_token_ids[batch_id][tok_id + chunk_overlap].item())

                all_y.append(y_resolved_list)
                all_y_hat.append(y_hat_resolved_list)
                all_tags.append(tags)
                all_predicted.append(predtags)
                all_words.append(vocab.tokenizer.convert_ids_to_tokens(token_list))
                all_token_ids.append(token_list)

        ## calc metric
        y_true = numpy.array(list(chain(*all_y)))
        y_pred = numpy.array(list(chain(*all_y_hat)))
        all_token_ids = numpy.array(list(chain(*all_token_ids)))

        num_proposed = len(y_pred[(vocab.OUTSIDE_ID > y_pred) & (all_token_ids > 0)])
        num_correct = (((y_true == y_pred) & (vocab.OUTSIDE_ID > y_true) & (all_token_ids > 0))).astype(numpy.int).sum()
        num_gold = len(y_true[(vocab.OUTSIDE_ID > y_true) & (all_token_ids > 0)])

        new_metrics = Metrics(
            epoch=epoch, step=step, num_correct=num_correct, num_proposed=num_proposed, num_gold=num_gold,
        )

        if save_predictions:
            final = args.logdir + "/%s.P%.2f_R%.2f_F%.2f" % (
                "{}-{}".format(str(epoch), str(step)),
                new_metrics.precision,
                new_metrics.recall,
                new_metrics.f1,
            )
            with open(final, "w") as fout:

                for words, tags, y_hat, preds in zip(all_words, all_tags, all_y_hat, all_predicted):
                    assert len(preds) == len(words) == len(tags)
                    for w, t, p in zip(words, tags, preds):
                        fout.write(f"{w}\t{t}\t{p}\n")
                    fout.write("\n")

                fout.write(f"num_proposed:{num_proposed}\n")
                fout.write(f"num_correct:{num_correct}\n")
                fout.write(f"num_gold:{num_gold}\n")
                fout.write(f"precision={new_metrics.precision}\n")
                fout.write(f"recall={new_metrics.recall}\n")
                fout.write(f"f1={new_metrics.f1}\n")

        if not args.dont_save_checkpoints:

            if save_checkpoint and metrics.was_improved(new_metrics):
                config = {
                    "args": args,
                    "optimizer_dense": optimizers[0].state_dict(),
                    "optimizer_sparse": optimizers[1].state_dict(),
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "performance": new_metrics.dict(),
                }
                fname = os.path.join(args.logdir, "{}-{}".format(str(epoch), str(step)))
                torch.save(config, f"{fname}.pt")
                fname = os.path.join(args.logdir, new_metrics.get_best_checkpoint_filename())
                torch.save(config, f"{fname}.pt")
                logging.info(f"weights were saved to {fname}.pt")

        if save_csv:
            new_metrics.to_csv(epoch=epoch, step=step, args=args)

        if metrics.was_improved(new_metrics):
            metrics.update(new_metrics)

        logging.info("Finished evaluation")

        return metrics

    def get_optimizers(self, args, checkpoint):

        optimizers = list()

        if args.encoder_lr > 0:
            optimizer_encoder = optim.Adam(
                list(self.bert.parameters()) + list(self.fc.parameters() if args.project else list()),
                lr=args.encoder_lr,
            )
            if args.resume_from_checkpoint is not None:
                optimizer_encoder.load_state_dict(checkpoint["optimizer_dense"])
                optimizer_encoder.param_groups[0]["lr"] = args.encoder_lr
                optimizer_encoder.param_groups[0]["weight_decay"] = args.encoder_weight_decay
            optimizers.append(optimizer_encoder)
        else:
            optimizers.append(DummyOptimizer(self.out.parameters(), defaults={}))

        if args.decoder_lr > 0:
            if args.sparse:
                optimizer_decoder = optim.SparseAdam(self.out.parameters(), lr=args.decoder_lr)
            else:
                optimizer_decoder = optim.Adam(self.out.parameters(), lr=args.decoder_lr)
            if args.resume_from_checkpoint is not None:
                optimizer_decoder.load_state_dict(checkpoint["optimizer_sparse"])
                if "weight_decay" not in optimizer_decoder.param_groups[0]:
                    optimizer_decoder.param_groups[0]["weight_decay"] = 0
                optimizer_decoder.param_groups[0]["lr"] = args.decoder_lr
                if not args.sparse:
                    optimizer_decoder.param_groups[0]["weight_decay"] = args.decoder_weight_decay
            optimizers.append(optimizer_decoder)
        else:
            optimizers.append(DummyOptimizer(self.out.parameters(), defaults={}))

        lr_schedulers = [
            getattr(LRSchedulers, lr_scheduler)(optimizer=optimizer, **lr_scheduler_config)
            for optimizer, (lr_scheduler, lr_scheduler_config) in zip(
                optimizers,
                [
                    (args.encoder_lr_scheduler, args.encoder_lr_scheduler_config),
                    (args.decoder_lr_scheduler, args.decoder_lr_scheduler_config),
                ],
            )
            if lr_scheduler is not None  # and not isinstance(optimizer, DummyOptimizer)
        ]

        return tuple(optimizers), tuple(lr_schedulers)
