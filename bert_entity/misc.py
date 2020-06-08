import argparse
import subprocess
from collections import Counter

import torch.optim
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


def capitalize(text: str) -> str:
    return text[0].upper() + text[1:]


def snip(string, search, keep, keep_search):
    pos = string.find(search)
    if pos != -1:
        if keep == "left":
            if keep_search:
                pos += len(search)
            string = string[:pos]
        if keep == "right":
            if not keep_search:
                pos += len(search)
            string = string[pos:]
    return string


def snip_anchor(text: str) -> str:
    return snip(text, "#", keep="left", keep_search=False)


def normalize_wiki_entity(i, replace_ws=False):
    i = snip_anchor(i)
    if len(i) == 0:
        return None
    i = capitalize(i)
    if replace_ws:
        return i.replace(" ", "_")
    return i


# most frequent English words from English Wikipedia
stopwords = {
    "a",
    "also",
    "an",
    "are",
    "as",
    "at",
    "be",
    "by",
    "city",
    "company",
    "film",
    "first",
    "for",
    "from",
    "had",
    "has",
    "her",
    "his",
    "in",
    "is",
    "its",
    "john",
    "national",
    "new",
    "of",
    "on",
    "one",
    "people",
    "school",
    "state",
    "the",
    "their",
    "these",
    "this",
    "time",
    "to",
    "two",
    "university",
    "was",
    "were",
    "with",
    "world",
}


def get_stopwordless_token_set(s):
    result = set(s.lower().split(" "))
    result_minus_stopwords = result.difference(stopwords)
    if len(result_minus_stopwords) == 0:
        return result
    else:
        return result_minus_stopwords


def argparse_bool_type(v):
    "Type for argparse that correctly treats Boolean values"
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], encoding="utf-8"
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def create_chunks(a_list, n):
    for i in range(0, len(a_list), n):
        yield a_list[i : i + n]


def unescape(s):
    if s.startswith('"'):
        s = s[1:-1]
    return s.replace('""""', '"').replace('""', '"')


def create_overlapping_chunks(a_list, n, overlap):
    for i in range(0, len(a_list), n - overlap):
        yield a_list[i : i + n]


def running_mean(new, old=None, momentum=0.9):
    if old is None:
        return new
    else:
        return momentum * old + (1 - momentum) * new


def get_topk_ids_aggregated_from_seq_prediction(logits, topk_per_token, topk_from_batch):
    topk_logit_per_token, topk_eids_per_token = logits.topk(topk_per_token, sorted=False, dim=-1)

    i = torch.cat(
        [
            topk_eids_per_token.view(1, -1),
            torch.zeros(topk_eids_per_token.view(-1).size(), dtype=torch.long, device=topk_eids_per_token.device).view(
                1, -1
            ),
        ],
        dim=0,
    )
    v = topk_logit_per_token.view(-1)
    st = torch.sparse.FloatTensor(i, v)
    stc = st.coalesce()
    topk_indices = stc._values().sort(descending=True)[1][:topk_from_batch]
    result = stc._indices()[0, topk_indices]

    return result.cpu().tolist()


def get_entity_annotations(t, outside_id):
    annos = list()
    begin = -1
    in_entity = -1
    for i, j in enumerate(t):
        if j < outside_id and begin == -1:
            begin = i
            in_entity = j.item()
        elif j < outside_id and j != in_entity:
            annos.append((tuple(range(begin, i)), in_entity))
            begin = i
            in_entity = j.item()
        elif j == outside_id and begin != -1:
            annos.append((tuple(range(begin, i)), in_entity))
            begin = -1
    return annos


def get_entity_annotations_with_gold_spans(t, t_gold, outside_id):
    annos = list()
    begin = -1
    in_gold_entity = -1
    collected_entities_in_span = Counter()
    for i, (j, j_gold) in enumerate(zip(t, t_gold)):
        if j_gold < outside_id and begin == -1:
            begin = i
            in_gold_entity = j_gold.item()
            collected_entities_in_span[j.item()] += 1
        elif j_gold != in_gold_entity and begin != -1:
            in_entity = collected_entities_in_span.most_common()[0][0]
            annos.append((tuple(range(begin, i)), in_entity))
            collected_entities_in_span = Counter()
            begin = i
            in_gold_entity = j_gold.item()
            collected_entities_in_span[j.item()] += 1
        elif j_gold == outside_id and begin != -1:
            in_entity = collected_entities_in_span.most_common()[0][0]
            annos.append((tuple(range(begin, i)), in_entity))
            collected_entities_in_span = Counter()
            begin = -1
    return annos


class DummyOptimizer(torch.optim.Optimizer):
    def step(self, closure=None):
        pass


class LRMilestones(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma once the number of epoch reaches one of the milestones. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = LRMilestones(optimizer, milestones=[(30, 0.1), (80, 0.2), ])
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, milestones, last_epoch=-1):
        super().__init__(optimizer)
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a list of" " increasing integers. Got {}", milestones)
        self.milestones = milestones
        super(LRMilestones, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        for ep, lr in self.milestones:
            if self.last_epoch >= ep:
                print("Set lr to {} in epoch {}".format(lr, ep))
                return lr


def pad_to(arr, max_len, pad_id, cls_id, sep_id):
    return [cls_id] + arr + [sep_id] + [pad_id] * (max_len - len(arr) - 2)


def set_out_id(t, repl, dummy=-1):
    t[(t == dummy)] = repl
    return t


class LRSchedulers:
    ReduceLROnPlateau = ReduceLROnPlateau
    LRMilestones = LRMilestones
