import pickle
from collections import OrderedDict

import numpy
import torch
from torch.utils import data

from misc import pad_to, set_out_id
from vocab import Vocab



class CONLLEDLDataset(data.Dataset):
    def __init__(self, args, split, vocab, device, label_size=None):

        if split == "train":
            train_valid_test_int = 0
        if split == "small_valid" or split == "valid":
            train_valid_test_int = 1
        if split == "test":
            train_valid_test_int = 2

        chunk_len = args.create_integerized_training_instance_text_length
        chunk_overlap = args.create_integerized_training_instance_text_overlap

        self.item_locs = None
        self.device = device
        with open(args.data_path_conll, "rb") as f:
            train_valid_test = pickle.load(f)
            self.conll_docs = torch.LongTensor(
                [
                    [
                        pad_to(
                            [tok_id for _, tok_id, _, _, _, _, _ in doc],
                            max_len=chunk_len + 2,
                            pad_id=0,
                            cls_id=101,
                            sep_id=102,
                        )
                        for doc in train_valid_test[train_valid_test_int]
                    ],
                    [
                        pad_to(
                            [bio_id for _, _, _, bio_id, _, _, _ in doc],
                            max_len=chunk_len + 2,
                            pad_id=2,
                            cls_id=2,
                            sep_id=2,
                        )
                        for doc in train_valid_test[train_valid_test_int]
                    ],
                    [
                        pad_to(
                            [wiki_id for _, _, _, _, _, wiki_id, _ in doc],
                            max_len=chunk_len + 2,
                            pad_id=vocab.PAD_ID,
                            cls_id=vocab.PAD_ID,
                            sep_id=vocab.PAD_ID,
                        )
                        for doc in train_valid_test[train_valid_test_int]
                    ],
                    [
                        pad_to(
                            [doc_id for _, _, _, _, _, _, doc_id in doc],
                            max_len=chunk_len + 2,
                            pad_id=0,
                            cls_id=0,
                            sep_id=0,
                        )
                        for doc in train_valid_test[train_valid_test_int]
                    ],
                ]
            ).permute(1, 0, 2)
            self.conll_docs[:, 2] = set_out_id(self.conll_docs[:, 2], vocab.OUTSIDE_ID)

        self.pad_token_id = vocab.PAD_ID
        self.label_size = label_size
        self.labels = None
        self.train_valid_test_int = train_valid_test_int

    def get_data_iter(
        self, args, batch_size, vocab, train,
    ):
        return data.DataLoader(
            dataset=self.conll_docs,
            batch_size=batch_size,
            shuffle=train,
            num_workers=args.data_workers,
            collate_fn=self.collate_func(
                args,
                return_labels=args.collect_most_popular_labels_steps is not None
                and args.collect_most_popular_labels_steps > 0
                if train
                else True,
                vocab=vocab,
            ),
        )

    def collate_func(self, args, vocab, return_labels):
        def collate(batch):
            return CONLLEDLDataset_collate_func(
                batch=batch,
                labels_with_high_model_score=self.labels,
                args=args,
                return_labels=return_labels,
                vocab=vocab,
                is_training=self.train_valid_test_int == 0,
            )

        return collate


def CONLLEDLDataset_collate_func(
    batch, labels_with_high_model_score, args, return_labels, vocab: Vocab, is_training=False,
):
    drop_entity_mentions_prob = args.maskout_entity_prob
    # print([b[0] for b in batch])
    label_size = args.label_size
    batch_token_ids = torch.LongTensor([b[0].tolist() for b in batch])
    batch_bio_ids = [b[1].tolist() for b in batch]
    batch_entity_ids = [b[2].tolist() for b in batch]
    batch_doc_ids = [b[3, 0].item() for b in batch]

    if return_labels:

        all_batch_entity_ids = OrderedDict()

        for batch_offset, one_item_entity_ids in enumerate(batch_entity_ids):
            for tok_id, eid in enumerate(one_item_entity_ids):
                if eid not in all_batch_entity_ids:
                    all_batch_entity_ids[eid] = len(all_batch_entity_ids)

        if label_size is not None:

            batch_shared_label_ids = all_batch_entity_ids.keys()
            negative_samples = set()
            if labels_with_high_model_score is not None:
                # print(labels_with_high_model_score)
                negative_samples = set(labels_with_high_model_score)
            # else:
            #     negative_samples = set(numpy.random.choice(vocab.OUTSIDE_ID, label_size, replace=False))
            if len(negative_samples) < label_size:
                random_negative_samples = set(numpy.random.choice(vocab.OUTSIDE_ID, label_size, replace=False))
                negative_samples = negative_samples.union(random_negative_samples)

            negative_samples.difference_update(batch_shared_label_ids)

            if len(batch_shared_label_ids) + len(negative_samples) < label_size:
                negative_samples.difference_update(
                    set(numpy.random.choice(vocab.OUTSIDE_ID, label_size, replace=False))
                )

            batch_shared_label_ids = (list(batch_shared_label_ids) + list(negative_samples))[:label_size]
            label_probs = torch.zeros(batch_token_ids.size(0), batch_token_ids.size(1), len(batch_shared_label_ids))
            bio_probs = torch.zeros(batch_token_ids.size(0), batch_token_ids.size(1), 3)

        else:

            batch_shared_label_ids = list(all_batch_entity_ids.keys())
            label_probs = torch.zeros(batch_token_ids.size(0), batch_token_ids.size(1), args.vocab_size)
            bio_probs = torch.zeros(batch_token_ids.size(0), batch_token_ids.size(1), 3)

        drop_probs = None
        if drop_entity_mentions_prob > 0 and is_training:
            drop_probs = torch.rand((batch_token_ids.size(0), batch_token_ids.size(1)),) < drop_entity_mentions_prob

        for batch_offset, (one_item_entity_ids, one_item_bio_ids) in enumerate(zip(batch_entity_ids, batch_bio_ids)):
            for tok_id, one_entity_ids in enumerate(one_item_entity_ids):

                if (
                    is_training
                    and vocab.OUTSIDE_ID != one_entity_ids
                    and drop_entity_mentions_prob > 0
                    and drop_probs[batch_offset][tok_id].item() == 1
                ):
                    batch_token_ids[batch_offset][tok_id] = vocab.tokenizer.vocab["[MASK]"]

                if label_size is not None:
                    label_probs[batch_offset][tok_id][torch.LongTensor([all_batch_entity_ids[one_entity_ids]])] = 1.0
                else:
                    label_probs[batch_offset][tok_id][torch.LongTensor(one_entity_ids)] = 1.0
                bio_probs[batch_offset][tok_id][torch.LongTensor(one_item_bio_ids)] = 1.0

        label_ids = torch.LongTensor(batch_shared_label_ids)

        return (
            batch_token_ids,
            label_ids,
            torch.LongTensor(batch_bio_ids),
            torch.FloatTensor(label_probs),
            bio_probs,
            None,
            {v: k for k, v in all_batch_entity_ids.items()},
            batch_entity_ids,
            batch_doc_ids,
            batch,
        )

    else:

        return batch_token_ids, None, None, None, None, None, None, None, None, batch

