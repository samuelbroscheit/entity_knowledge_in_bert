import os
from collections import OrderedDict

import pandas
from itertools import cycle
from operator import gt, lt


class Metrics:

    meta = OrderedDict(
        [
            ("epoch", {"comp": gt, "type": int, "str": lambda a: a}),
            ("step", {"comp": gt, "type": int, "str": lambda a: a}),
            ("num_gold", {"comp": gt, "type": int, "str": lambda a: a}),
            ("num_correct", {"comp": gt, "type": int, "str": lambda a: a}),
            ("num_proposed", {"comp": gt, "type": int, "str": lambda a: a}),
            ("f1", {"comp": gt, "type": float, "str": lambda a: f"{a:.5f}"}),
            ("f05", {"comp": gt, "type": float, "str": lambda a: f"{a:.5f}"}),
            ("precision", {"comp": gt, "type": float, "str": lambda a: f"{a:.5f}"}),
            ("recall", {"comp": gt, "type": float, "str": lambda a: f"{a:.5f}"}),
            ("span_f1", {"comp": gt, "type": float, "str": lambda a: f"{a:.5f}"}),
            ("span_precision", {"comp": gt, "type": float, "str": lambda a: f"{a:.5f}"}),
            ("span_recall", {"comp": gt, "type": float, "str": lambda a: f"{a:.5f}"}),
            ("lenient_span_f1", {"comp": gt, "type": float, "str": lambda a: f"{a:.5f}"}),
            ("lenient_span_precision", {"comp": gt, "type": float, "str": lambda a: f"{a:.5f}"}),
            ("lenient_span_recall", {"comp": gt, "type": float, "str": lambda a: f"{a:.5f}"}),
            ("precision_gold_mentions", {"comp": gt, "type": float, "str": lambda a: f"{a:.5f}"}),
            ("avg_loss", {"comp": lt, "type": float, "str": lambda a: f"{a:.5f}"}),
        ]
    )

    def __init__(self, epoch=0, step=0, num_correct=0, num_gold=0, num_proposed=0, model_selection="f1", num_best_checkpoints=4, **kwargs):

        self.epoch = epoch
        self.step = step
        self.num_correct = num_correct
        self.num_gold = num_gold
        self.num_proposed = num_proposed

        self.precision = Metrics.compute_precision(num_correct, num_proposed)
        self.recall = Metrics.compute_recall(num_correct, num_gold)
        self.f1 = Metrics.compute_fmeasure(self.precision, self.recall)
        self.f05 = Metrics.compute_fmeasure(self.precision, self.recall, weight=1.5)

        self.avg_loss = float("inf")

        for k,v in kwargs.items():
            if k in self.meta:
                self.__dict__[k] = v

        self.model_selection = model_selection
        self.checkpoint_cycle = cycle(range(num_best_checkpoints),)

    @staticmethod
    def compute_precision(correct, proposed):
        try:
            precision = correct/proposed
        except ZeroDivisionError:
            precision = 0.0
        return precision

    @staticmethod
    def compute_recall(correct, gold):
        try:
            recall = correct/ gold
        except ZeroDivisionError:
            recall = 0.0
        return recall

    @staticmethod
    def compute_fmeasure(precision, recall, weight=2.0):
        try:
            f = weight * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f = 0.0
        return f

    def was_improved(self, other: "Metrics"):
        return Metrics.meta[self.model_selection]["comp"](
            other.get_model_selection_metric(), self.get_model_selection_metric()
        )

    def update(self, other: "Metrics"):
        if self.was_improved(other):
            for key, val in other.__dict__.items():
                self.__setattr__(key, other.__dict__.get(key))

    def get_model_selection_metric(self):
        return self.__dict__.get(self.model_selection)

    def get_best_checkpoint_filename(self):
        return f"best_{self.model_selection}-{next(self.checkpoint_cycle)}"

    def to_csv(self, epoch, step, args):
        header = (
            [k for k in list(self.meta.keys()) if k in self.__dict__]
            if not os.path.exists("{}/log.csv".format(args.logdir))
            else False
        )
        pandas.DataFrame(
            [[self.__dict__[k] for k in list(self.meta.keys()) if k in self.__dict__]]
        ).to_csv(f"{args.logdir}/log.csv", mode="a", header=header)

    def dict(self):
        return self.__dict__

    def __repr__(self):
        return str(self.__dict__)

    def report(self, filter=None):
        if not filter:
            filter = set(self.meta.keys())
        return [f"{k}: {Metrics.meta[k]['str'](self.__dict__[k])}" for k in list(self.meta.keys()) if k in filter and k in self.__dict__]
