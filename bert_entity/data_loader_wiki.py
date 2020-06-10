import os
import pickle
from collections import OrderedDict
from operator import itemgetter

import numpy
import torch
from torch.utils import data
from tqdm import tqdm

from vocab import Vocab


class EDLDataset(data.Dataset):
    def __init__(self, args, split, vocab, device, label_size=None):

        if split == "train":
            loc_file_name = args.train_loc_file
            self.data_dir = args.train_data_dir
        elif split == "valid":
            loc_file_name = args.valid_loc_file
            self.data_dir = args.valid_data_dir
        elif split == "test":
            loc_file_name = args.test_loc_file
            self.data_dir = args.test_data_dir


        self.data_path = f"data/versions/{args.data_version_name}/wiki_training/integerized/{args.wiki_lang_version}/"
        self.item_locs = None
        self.device = device
        if os.path.exists("{}.pickle".format(self.data_path + loc_file_name)):
            with open("{}.pickle".format(self.data_path + loc_file_name), "rb") as f:
                self.item_locs = pickle.load(f)
        else:
            with open(self.data_path + loc_file_name) as f:
                self.item_locs = list(map(lambda x: list(map(int, x.strip().split())), tqdm(f.readlines())))
            with open("{}.pickle".format(self.data_path + loc_file_name), "wb") as f:
                pickle.dump(self.item_locs, f)
        self.pad_token_id = vocab.PAD_ID
        self.label_size = label_size
        self.is_training = split == "train"

    def get_data_iter(
        self, args, batch_size, vocab, train,
    ):
        return data.DataLoader(
            dataset=self.item_locs,
            batch_size=batch_size,
            shuffle=train,
            num_workers=args.data_workers,
            collate_fn=self.collate_func(
                args=args,
                vocab=vocab,
                return_labels=args.collect_most_popular_labels_steps is not None
                and args.collect_most_popular_labels_steps > 0
                if train
                else True,
            ),
        )

    # def collate_func(self, args, vocab, return_labels, shards, shards_locks):
    def collate_func(
        self, args, vocab, return_labels, in_queue=None, out_queue=None,
    ):
        def collate(batch):
            return EDLDataset_collate_func(
                batch=batch,
                labels_with_high_model_score=None,
                args=args,
                return_labels=return_labels,
                data_path=self.data_path,
                vocab=vocab,
                is_training=self.is_training,
            )

        return collate


def EDLDataset_collate_func(
    batch,
    labels_with_high_model_score,
    args,
    return_labels,
    vocab: Vocab,
    data_path=None,
    is_training=True,
    drop_entity_mentions_prob=0.0,
    loaded_batch=None,
):
    if loaded_batch is None:
        batch_dict_list = list()
        for shard, offset in batch:
            # print('{}/{}.dat'.format(data_path, shard), offset)
            with open("{}/{}.dat".format(data_path, shard), "rb") as f:
                f.seek(offset)
                (
                    token_ids_chunk,
                    mention_entity_ids_chunk,
                    mention_entity_probs_chunk,
                    mention_probs_chunk,
                ) = pickle.load(f)
                try:
                    eval_mask = list(map(is_a_wikilink_or_keyword, mention_probs_chunk))
                    mention_entity_ids_chunk = list(map(itemgetter(0), mention_entity_ids_chunk))
                    mention_entity_probs_chunk = list(map(itemgetter(0), mention_entity_probs_chunk))
                    batch_dict_list.append(
                        {
                            "token_ids": token_ids_chunk,
                            "entity_ids": mention_entity_ids_chunk,
                            "entity_probs": mention_entity_probs_chunk,
                            "eval_mask": eval_mask,
                        }
                    )
                except Exception as e:
                    print(f"pickle.load(shards[shard]) failed {e}")
                    print(mention_entity_ids_chunk)
                    print(mention_entity_probs_chunk)
                    raise e

        f = lambda x: [sample[x] for sample in batch_dict_list]
        # print(batch)
        batch_token_ids = f("token_ids")
        batch_entity_ids = f("entity_ids")
        batch_entity_probs = f("entity_probs")
        eval_mask = f("eval_mask")
        maxlen = max([len(chunk) for chunk in batch_token_ids])

        eval_mask = torch.LongTensor([sample + [0] * (maxlen - len(sample)) for sample in eval_mask])

        # create dictionary mapping the vocabulary entity id to a batch label id
        #
        # e.g.
        # all_batch_entity_ids[324] = 0
        # all_batch_entity_ids[24]  = 1
        # all_batch_entity_ids[2]   = 2
        # all_batch_entity_ids[987] = 3
        #
        all_batch_entity_ids = OrderedDict()

        for batch_offset, (batch_item_token_item_entity_ids, batch_item_token_entity_probs) in enumerate(
            zip(batch_entity_ids, batch_entity_probs)
        ):
            for tok_id, (token_entity_ids, token_entity_probs) in enumerate(
                zip(batch_item_token_item_entity_ids, batch_item_token_entity_probs)
            ):
                for eid in token_entity_ids:
                    if eid not in all_batch_entity_ids:
                        all_batch_entity_ids[eid] = len(all_batch_entity_ids)

        loaded_batch = (
            batch_token_ids,
            batch_entity_ids,
            batch_entity_probs,
            eval_mask,
            all_batch_entity_ids,
            maxlen,
        )

    else:
        (batch_token_ids, batch_entity_ids, batch_entity_probs, eval_mask, all_batch_entity_ids, maxlen,) = loaded_batch

    batch_token_ids = torch.LongTensor([sample + [0] * (maxlen - len(sample)) for sample in batch_token_ids])

    if return_labels:

        # if labels for each token should be over
        # a. the whole entity vocabulary
        # b. a reduced set of entities composed of:
        #       set of batch's true entities, entities
        #       set of entities with the largest logits
        #       set of negative samples

        if args.label_size is None:

            batch_shared_label_ids = list(all_batch_entity_ids.keys())
            label_probs = torch.zeros(batch_token_ids.size(0), batch_token_ids.size(1), args.vocab_size)

        else:

            # batch_shared_label_ids are constructing by incrementally concatenating
            #       set of batch's true entities, entities
            #       set of entities with the largest logits
            #       set of negative samples

            batch_shared_label_ids = list(all_batch_entity_ids.keys())

            if len(batch_shared_label_ids) < args.label_size and labels_with_high_model_score is not None:
                # print(labels_with_high_model_score)
                negative_examples = set(labels_with_high_model_score)
                negative_examples.difference_update(batch_shared_label_ids)
                batch_shared_label_ids += list(negative_examples)

            if len(batch_shared_label_ids) < args.label_size:
                negative_samples = set(numpy.random.choice(vocab.OUTSIDE_ID, args.label_size, replace=False))
                negative_samples.difference_update(batch_shared_label_ids)
                batch_shared_label_ids += list(negative_samples)

            batch_shared_label_ids = batch_shared_label_ids[: args.label_size]

            label_probs = torch.zeros(batch_token_ids.size(0), batch_token_ids.size(1), len(batch_shared_label_ids))

        drop_probs = None
        if drop_entity_mentions_prob > 0 and is_training:
            drop_probs = torch.rand((batch_token_ids.size(0), batch_token_ids.size(1)),) < drop_entity_mentions_prob

        # loop through the batch x tokens x (label_ids, label_probs)
        for batch_offset, (batch_item_token_item_entity_ids, batch_item_token_entity_probs) in enumerate(
            zip(batch_entity_ids, batch_entity_probs)
        ):
            # loop through tokens x (label_ids, label_probs)
            for tok_id, (token_entity_ids, token_entity_probs) in enumerate(
                zip(batch_item_token_item_entity_ids, batch_item_token_entity_probs)
            ):
                if drop_entity_mentions_prob > 0 and is_training and drop_probs[batch_offset][tok_id].item() == 1:
                    batch_token_ids[batch_offset][tok_id] = vocab.tokenizer.vocab["[MASK]"]

                if args.label_size is None:
                    label_probs[batch_offset][tok_id][torch.LongTensor(token_entity_ids)] = torch.Tensor(
                        batch_item_token_item_entity_ids
                    )
                else:
                    label_probs[batch_offset][tok_id][
                        torch.LongTensor(list(map(all_batch_entity_ids.__getitem__, token_entity_ids)))
                    ] = torch.Tensor(token_entity_probs)

        label_ids = torch.LongTensor(batch_shared_label_ids)

        return (
            batch_token_ids,
            label_ids,
            label_probs,
            torch.LongTensor(eval_mask),
            {v: k for k, v in all_batch_entity_ids.items()},
            batch_entity_ids,
            batch,
            loaded_batch,
        )

    else:

        return batch_token_ids, None, None, None, None, None, batch, loaded_batch

# hack to detect if an entity annotation was a
# wikilink (== only one entity label) or a
# keyword matcher annotation (== multiple entity labels)
def is_a_wikilink_or_keyword(item):
    if len(item) == 1:
        return 1
    else:
        return 0
