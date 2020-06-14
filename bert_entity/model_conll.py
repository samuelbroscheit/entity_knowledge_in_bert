import logging
import os
import re

import numpy
import torch
import torch.nn as nn
from torch import optim
from tqdm import trange, tqdm

from metrics import Metrics
from data_loader_conll import CONLLEDLDataset_collate_func
from misc import (
    running_mean,
    get_entity_annotations,
    get_entity_annotations_with_gold_spans,
    DummyOptimizer,
    LRSchedulers,
    create_overlapping_chunks,
    get_topk_ids_aggregated_from_seq_prediction,
)
from pytorch_pretrained_bert import BertModel

bio_id = {
    "B": 0,
    "I": 1,
    "O": 2,
}
bio_id_inv = {
    0: "B",
    1: "I",
    2: "O",
}


class ConllNet(nn.Module):
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

        self.out_segm = nn.Sequential(nn.Dropout(args.bert_dropout), nn.Linear(768, 768), nn.Tanh(), nn.Linear(768, 3),)

        # torch.nn.init.normal_(self.out, std=0.1)

        if args.bert_dropout and args.bert_dropout > 0:
            for m in self.bert.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = args.bert_dropout

        self.device = args.device
        self.out_device = args.out_device
        self.finetuning = args.finetuning
        self.vocab_size = vocab_size

    def to(self, device, out_device):
        self.bert.to(device)
        if self.fc:
            self.fc.to(device)
        self.out.to(out_device)
        self.out_segm.to(device)
        self.device = device
        self.out_device = out_device

    def forward(self, x, y=None, probs=None, segm_probs=None, enc=None):
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
        if segm_probs is not None:
            segm_probs = segm_probs.to(self.out_device)

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

        logits_segm = self.out_segm(enc)

        if y is not None:
            out = self.out(y)
            logits = enc.matmul(out.transpose(0, 1))
            y_hat = logits.argmax(-1)
            bio_y_hat = logits_segm.argmax(-1)
            return logits, y, y_hat, probs, segm_probs, out, enc, logits_segm, bio_y_hat
        else:
            with torch.no_grad():
                out = self.out.weight
                logits = enc.matmul(out.transpose(0, 1))
                y_hat = logits.argmax(-1)
                bio_y_hat = logits_segm.argmax(-1)
                return logits, None, y_hat, None, None, None, enc, None, bio_y_hat

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

        with trange(len(train_iter)) as t:
            for iter, batch in enumerate(train_iter):

                model.to(
                    args.device, args.out_device,
                )
                model.train()

                (
                    batch_token_ids,
                    label_ids,
                    _,
                    label_probs,
                    batch_bio_probs,
                    _,
                    label_id_to_entity_id_dict,
                    batch_entity_ids,
                    batch_doc_ids,
                    orig_batch,
                ) = batch

                enc = None

                labels_with_high_model_score = list()
                if (
                    args.collect_most_popular_labels_steps is not None
                    and args.collect_most_popular_labels_steps > 0
                    and iter > 0
                    and iter % args.collect_most_popular_labels_steps == 0
                ):
                    model.to(args.device, args.eval_device)
                    logits, _, y_hat, _, _, _, enc, segm_logits, segm_pred = model(
                        batch_token_ids, None, None, batch_bio_probs
                    )  # logits: (N, T, VOCAB), y: (N, T)
                    labels_with_high_model_score = get_topk_ids_aggregated_from_seq_prediction(
                        logits, topk_from_batch=args.label_size, topk_per_token=args.topk_neg_examples
                    )
                    (
                        batch_token_ids,
                        label_ids,
                        _,
                        label_probs,
                        batch_bio_probs,
                        _,
                        label_id_to_entity_id_dict,
                        batch_entity_ids,
                        batch_doc_ids,
                        orig_batch,
                    ) = CONLLEDLDataset_collate_func(
                        args=args,
                        labels_with_high_model_score=labels_with_high_model_score,
                        batch=orig_batch,
                        return_labels=True,
                        vocab=vocab,
                    )

                # if args.label_size is not None:
                logits, y, y_hat, label_probs, batch_bio_probs, sparse_params, _, segm_logits, segm_pred = model(
                    batch_token_ids, label_ids, label_probs, batch_bio_probs, enc=enc
                )  # logits: (N, T, VOCAB), y: (N, T)
                # else:
                #     logits, y, y_hat, label_probs, sparse_params = model(batch_token_ids, None, label_probs) # logits: (N, T, VOCAB), y: (N, T)

                # logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
                logits = logits.view(-1)  # (N*T, VOCAB)
                segm_logits = segm_logits.view(-1)  # (N*T, VOCAB)
                label_probs = label_probs.view(-1)  # (N*T,)
                batch_bio_probs = batch_bio_probs.view(-1)

                task_importance_ratio = 0.1

                if args.learn_segmentation:
                    loss = (1 - task_importance_ratio) * criterion(
                        logits, label_probs
                    ) + task_importance_ratio * criterion(segm_logits, batch_bio_probs)
                else:
                    loss = criterion(logits, label_probs)

                loss.backward()

                if (iter + 1) % args.accumulate_batch_gradients == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()

                if iter == 0:
                    logging.debug("=====sanity check======")
                    logging.debug("x:", batch_token_ids.cpu().numpy()[0])
                    logging.debug("tokens:", vocab.tokenizer.convert_ids_to_tokens(batch_token_ids.cpu().numpy()[0]))
                    logging.debug("y:", label_probs.cpu().numpy()[0])
                    logging.debug("=======================")

                loss_aggr = running_mean(loss.detach().item(), loss_aggr)

                if iter > 0 and iter % args.checkpoint_eval_steps == 0:
                    metrics = ConllNet.evaluate(
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
                    # nr_labels=len(label_ids),
                    # aggr_labels=len(labels_with_high_model_score) if labels_with_high_model_score else 0,
                    last_eval=metrics.report(filter={"f1", "span_f1", "lenient_span_f1", "epoch", "step"}),
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

        logging.info(f"Start evaluation on split {'test' if args.eval_on_test_only else 'valid'}")

        model.eval()
        model.to(args.device, args.eval_device)

        chunk_len = args.create_integerized_training_instance_text_length
        chunk_overlap = args.create_integerized_training_instance_text_overlap

        all_words = list()
        all_tags = [0] * (len(iterator) * args.eval_batch_size * (chunk_len))
        all_y = [0] * (len(iterator) * args.eval_batch_size * (chunk_len))
        all_y_hat = [0] * (len(iterator) * args.eval_batch_size * (chunk_len))
        all_segm_preds = [0] * (len(iterator) * args.eval_batch_size * (chunk_len))
        all_y_hat_gold_mentions = [0] * (len(iterator) * args.eval_batch_size * (chunk_len))
        all_logits = [0] * (len(iterator) * args.eval_batch_size * (chunk_len))
        all_predicted = [0] * (len(iterator) * args.eval_batch_size * (chunk_len))
        all_token_ids = [0] * (len(iterator) * args.eval_batch_size * (chunk_len))

        all_y_hat_scores = torch.ones(len(iterator) * args.eval_batch_size * (chunk_len)) * -1e10
        all_y_hat_gold_mentions_scores = torch.ones(len(iterator) * args.eval_batch_size * (chunk_len)) * -1e10

        best_scores = torch.ones(len(iterator) * args.eval_batch_size * (chunk_len)) * -1e10

        offset = 0
        last_doc = -1

        # new_best_top1_logit = torch.ones((chunk_len))
        # new_best_top2_logit_gold_mentions = torch.ones((chunk_len))

        with torch.no_grad():

            for iter, batch in enumerate(tqdm(iterator)):

                (
                    batch_token_ids,
                    label_ids,
                    batch_bio_ids,
                    label_probs,
                    batch_bio_probs,
                    _,
                    label_id_to_entity_id_dict,
                    batch_entity_ids,
                    batch_doc_ids,
                    orig_batch,
                ) = batch
                eval_mask = batch_bio_ids == 2
                logits, y, y_hat, probs, batch_bio_probs, _, _, segm_logits, segm_preds = model(
                    batch_token_ids, None, label_probs, batch_bio_probs
                )  # logits: (N, T, VOCAB), y: (N, T)

                logits = logits[:, 1:-1, :]
                label_probs = label_probs[:, 1:-1, :]
                y_hat = y_hat[:, 1:-1]
                segm_preds = segm_preds[:, 1:-1]
                eval_mask = eval_mask[:, 1:-1]
                batch_token_ids = batch_token_ids[:, 1:-1]

                top2_logit, top2 = logits.topk(k=2, dim=-1,)
                top2_select = (y_hat >= vocab.OUTSIDE_ID).long()

                y_hat_gold_mentions = (
                    top2.view(-1, top2.size(-1)).gather(dim=1, index=top2_select.view(-1, 1)).view(y_hat.size())
                )
                top2_logit_gold_mentions = (
                    top2_logit.view(-1, top2_logit.size(-1))
                    .gather(dim=1, index=top2_select.view(-1, 1))
                    .view(y_hat.size())
                    .to("cpu")
                )

                top1_logit, _ = logits.to("cpu").max(dim=-1,)
                top1_probs = torch.sigmoid(top1_logit)

                for batch_id, seq in enumerate(label_probs.max(-1)[1]):

                    if last_doc >= 0:
                        if last_doc == batch_doc_ids[batch_id]:
                            next_step = chunk_len - chunk_overlap
                        else:
                            last_doc = batch_doc_ids[batch_id]
                            next_step = chunk_len

                        offset += next_step
                    else:
                        last_doc = batch_doc_ids[batch_id]

                    new_best_top1_logit = top1_logit[batch_id] > best_scores[offset : offset + chunk_len]
                    new_best_top2_logit_gold_mentions = (
                        top2_logit_gold_mentions[batch_id] > best_scores[offset : offset + chunk_len]
                    )

                    all_y_hat_scores[offset : offset + chunk_len] = (
                        new_best_top1_logit.float() * top1_logit[batch_id]
                        + (1.0 - new_best_top1_logit.float()) * all_y_hat_scores[offset : offset + chunk_len]
                    )
                    all_y_hat_gold_mentions_scores[offset : offset + chunk_len] = (
                        new_best_top2_logit_gold_mentions.float() * top1_logit[batch_id]
                        + (1.0 - new_best_top2_logit_gold_mentions.float())
                        * all_y_hat_gold_mentions_scores[offset : offset + chunk_len]
                    )

                    for tok_id, label_id in enumerate(seq):

                        y_resolved = (
                            label_ids[label_id].item() if eval_mask[batch_id][tok_id] == 0 else vocab.OUTSIDE_ID
                        )
                        all_y[offset + tok_id] = y_resolved
                        all_tags[offset + tok_id] = vocab.idx2tag[y_resolved]

                        y_hat_resolved = (
                            new_best_top1_logit[tok_id].item() * y_hat[batch_id][tok_id].item()
                            + (1 - new_best_top1_logit[tok_id].item()) * all_y_hat[offset + tok_id]
                        )
                        all_y_hat[offset + tok_id] = y_hat_resolved
                        all_predicted[offset + tok_id] = vocab.idx2tag[y_hat_resolved]

                        y_hat_gold_mentions_resolved = (
                            new_best_top2_logit_gold_mentions[tok_id].item()
                            * y_hat_gold_mentions[batch_id][tok_id].item()
                            + (1 - new_best_top2_logit_gold_mentions[tok_id].item()) * all_y_hat[offset + tok_id]
                        )
                        all_y_hat_gold_mentions[offset + tok_id] = y_hat_gold_mentions_resolved

                        all_segm_preds[offset + tok_id] = segm_preds[batch_id][tok_id].item()
                        all_token_ids[offset + tok_id] = batch_token_ids[batch_id][tok_id].item()
                        all_logits[offset + tok_id] = top1_probs[batch_id][tok_id].item()

        all_tags = all_tags[: offset + chunk_len]
        all_y = all_y[: offset + chunk_len]
        all_y_hat = all_y_hat[: offset + chunk_len]
        all_y_hat_gold_mentions = all_y_hat_gold_mentions[: offset + chunk_len]
        all_logits = all_logits[: offset + chunk_len]
        all_predicted = all_predicted[: offset + chunk_len]
        all_token_ids = all_token_ids[: offset + chunk_len]
        all_segm_preds = all_segm_preds[: offset + chunk_len]

        for chunk in create_overlapping_chunks(all_token_ids, 512, 0):
            all_words.extend(vocab.tokenizer.convert_ids_to_tokens(chunk))

        ## calc metric
        y_true = numpy.array(all_y)
        y_pred = numpy.array(all_y_hat)
        y_pred_gold_mentions = numpy.array(all_y_hat_gold_mentions)
        all_token_ids = numpy.array(all_token_ids)

        spans_true = get_entity_annotations(y_true, vocab.OUTSIDE_ID)
        spans_pred = get_entity_annotations(y_pred, vocab.OUTSIDE_ID)
        spans_pred_gold_mentions = get_entity_annotations_with_gold_spans(
            y_pred_gold_mentions, y_true, vocab.OUTSIDE_ID
        )

        overlaps = list()
        for anno in spans_pred:
            overlaps.extend(filter(lambda s: len(set(anno[0]) & set(s[0])) > 0 and anno[1] == s[1], spans_true))

        overlaps_gold_mentions = list()
        for anno in spans_pred_gold_mentions:
            overlaps_gold_mentions.extend(
                filter(lambda s: len(set(anno[0]) & set(s[0])) > 0 and anno[1] == s[1], spans_true)
            )
        num_lenient_correct_gold_mentions = len(set(overlaps_gold_mentions))

        num_proposed = len(y_pred[(vocab.OUTSIDE_ID > y_pred) & (all_token_ids > 0)])
        num_correct = (((y_true == y_pred) & (vocab.OUTSIDE_ID > y_true) & (all_token_ids > 0))).astype(numpy.int).sum()

        num_correct_gold_mentions = (
            (((y_true == y_pred_gold_mentions) & (vocab.OUTSIDE_ID > y_true) & (all_token_ids > 0)))
            .astype(numpy.int)
            .sum()
        )

        num_gold = len(y_true[(vocab.OUTSIDE_ID > y_true) & (all_token_ids > 0)])

        num_spans_correct = len(set(spans_true).intersection(set(spans_pred)))
        num_spans_true = len(set(spans_true))
        num_spans_proposed = len(set(spans_pred)) if len(set(spans_pred)) > 0 else 0

        num_lenient_correct_spans = len(set(overlaps))

        new_metrics = Metrics(
            epoch=epoch,
            step=step,
            num_correct=num_correct,
            num_gold=num_gold,
            num_proposed=num_proposed,
            # in this setting all gold mentions are scored which is why num_gold == num_proposed
            precision_gold_mentions=Metrics.compute_precision(correct=num_correct_gold_mentions, proposed=num_gold),
            span_precision=Metrics.compute_precision(correct=num_spans_correct, proposed=num_spans_proposed),
            span_recall=Metrics.compute_recall(correct=num_spans_correct, gold=num_spans_true),
            span_f1=Metrics.compute_fmeasure(
                precision=Metrics.compute_precision(correct=num_spans_correct, proposed=num_spans_proposed),
                recall=Metrics.compute_recall(correct=num_spans_correct, gold=num_spans_true),
            ),
            lenient_span_precision=Metrics.compute_precision(
                correct=num_lenient_correct_spans, proposed=num_spans_proposed
            ),
            lenient_span_recall=Metrics.compute_recall(correct=num_lenient_correct_spans, gold=num_spans_true),
            lenient_span_f1=Metrics.compute_fmeasure(
                precision=Metrics.compute_precision(correct=num_lenient_correct_spans, proposed=num_spans_proposed),
                recall=Metrics.compute_recall(correct=num_lenient_correct_spans, gold=num_spans_true),
            ),
        )

        if save_predictions:
            final = (
                args.logdir
                + "/{}-{}-MENTION-S_P_{:.2f}_R_{:.2f}_F1_{:.2f}-LS_P_{:.2f}_R_{:.2f}_F1_{:.2f}-T_P_{:.2f}_R_{:.2f}_F1_{:.2f}-LINK-S_P_{:.2f}.txt".format(
                    epoch,
                    step,
                    new_metrics.span_precision,
                    new_metrics.span_recall,
                    new_metrics.span_f1,
                    new_metrics.lenient_span_precision,
                    new_metrics.lenient_span_recall,
                    new_metrics.lenient_span_f1,
                    new_metrics.precision,
                    new_metrics.recall,
                    new_metrics.f1,
                    new_metrics.precision_gold_mentions,
                )
            )
            with open(final, "w") as fout:

                for words, tags, y_hat, preds, segm_pred, logits in zip(
                    all_words, all_tags, all_y_hat, all_predicted, all_segm_preds, all_logits
                ):
                    fout.write(f"{words}\t{tags}\t{preds}\t{bio_id_inv[segm_pred]}\t{logits}\n")

                fout.write(f"num_proposed:{new_metrics.num_proposed}\n")
                fout.write(f"num_correct:{new_metrics.num_correct}\n")
                fout.write(f"num_gold:{new_metrics.num_gold}\n")
                fout.write(f"precision={new_metrics.precision}\n")
                fout.write(f"precision_gold_mentions={new_metrics.precision_gold_mentions}\n")
                fout.write(f"recall={new_metrics.recall}\n")
                fout.write(f"f1={new_metrics.f1}\n")

        if not args.dont_save_checkpoints:
            if save_checkpoint or metrics.was_improved(new_metrics):
                config = {
                    "args": args,
                    "optimizer_dense": optimizers[0].state_dict() if optimizers else None,
                    "optimizer_sparse": optimizers[1].state_dict() if optimizers else None,
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
            if args.exclude_parameter_names_regex is not None:
                bert_parameters = list()
                regex = re.compile(args.exclude_parameter_names_regex)
                for n, p in list(self.bert.named_parameters()):
                    if not len(regex.findall(n)) > 0:
                        bert_parameters.append(p)
            else:
                bert_parameters = list(self.bert.parameters())
            optimizer_encoder = optim.Adam(
                bert_parameters + list(self.fc.parameters() if args.project else list()), lr=args.encoder_lr
            )
            # optimizer_encoder = BertAdam(bert_parameters + list(self.fc.parameters() if args.project else list()),
            #                      lr=args.encoder_lr,
            # )

            if args.resume_optimizer_from_checkpoint:
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

        if args.segm_decoder_lr > 0:
            optimizer_segm_decoder = optim.Adam(self.out_segm.parameters(), lr=args.segm_decoder_lr)
            if args.resume_optimizer_from_checkpoint:
                optimizer_segm_decoder.param_groups[0]["lr"] = args.segm_decoder_lr
                optimizer_segm_decoder.param_groups[0]["weight_decay"] = args.segm_decoder_weight_decay
            optimizers.append(optimizer_segm_decoder)
        else:
            optimizers.append(DummyOptimizer(self.out.parameters(), defaults={}))

        lr_schedulers = [
            getattr(LRSchedulers, lr_scheduler)(optimizer=optimizer, **lr_scheduler_config)
            for optimizer, (lr_scheduler, lr_scheduler_config) in zip(
                optimizers,
                [
                    (args.encoder_lr_scheduler, args.encoder_lr_scheduler_config),
                    (args.decoder_lr_scheduler, args.decoder_lr_scheduler_config),
                    (args.segm_decoder_lr_scheduler, args.segm_decoder_lr_scheduler_config),
                ],
            )
            if lr_scheduler is not None  # and not isinstance(optimizer, DummyOptimizer)
        ]

        return tuple(optimizers), tuple(lr_schedulers)
