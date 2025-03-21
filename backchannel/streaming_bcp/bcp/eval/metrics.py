"""
평가 지표 모듈
"""

import numpy as np

from itertools import chain
from sklearn import metrics  # f1 score 계산

import bcp.const.const as C


class Metric:
    def __init__(self, dataset, category_label, y_true, y_pred, verbose=False):
        self.y_true = y_true
        self.y_pred = y_pred

        self.verbose = verbose

        self.glabel_dict, self.label2idx = self.load_labels(dataset, category_label)
        self.y_true_ids = self.transform_labels(
            self.y_true, self.label2idx, self.glabel_dict
        )
        self.y_pred_ids = self.transform_labels(self.y_pred, self.label2idx)

    def load_labels(self, dataset, category_label):
        glabel_dict = {}
        label2idx = {}

        if dataset == "swbd":
            for x, y in C.SWBD_BC_CATEGORIES.items():
                if category_label == "merge":
                    glabel_dict.setdefault(x, y[0])
                    label2idx.setdefault(y[0], len(label2idx))
                elif category_label == "binary":
                    glabel_dict.setdefault(x, y[2])
                    label2idx.setdefault(y[2], len(label2idx))
                else:
                    glabel_dict.setdefault(x, y[1])
                    label2idx.setdefault(y[1], len(label2idx))
        else:
            for x, y in C.BC_CATEGORIES.items():
                if category_label == "merge":
                    glabel_dict.setdefault(x, y[0])
                    label2idx.setdefault(y[0], len(label2idx))
                elif category_label == "binary":
                    glabel_dict.setdefault(x, y[2])
                    label2idx.setdefault(y[2], len(label2idx))
                else:
                    glabel_dict.setdefault(x, y[1])
                    label2idx.setdefault(y[1], len(label2idx))

        if self.verbose:
            print(f"Ground Truth Label: {glabel_dict}")
            print(f"Label to Index: {label2idx}")

        return glabel_dict, label2idx

    def transform_labels(self, ys, label2idx, glabel_dict=None):
        y_ids = {}

        for k, y in ys.items():
            new_y = [
                label2idx[i] if glabel_dict is None else label2idx[glabel_dict[i]]
                for i in y
            ]
            y_ids.update({k: new_y})

        return y_ids

    def get_labels(self):
        return [
            x for x in sorted(self.label2idx.items(), key=lambda x: x[1], reverse=False)
        ]

    def get_label_indices(self):
        return sorted(list(self.label2idx.values()))

    def get_preds(self):
        return self.flatten(self.y_pred_ids)

    def get_golds(self):
        return self.flatten(self.y_true_ids)

    def flatten(self, datas):
        return np.array(list(chain(*datas.values())))

    def natural_score(self, score):
        if isinstance(score, np.ndarray):
            return [round(s.item() * 100.0, 2) for s in score]
        else:
            return round(score * 100.0, 2)

    def macrof1_score(self, yt, y):
        return self.natural_score(
            metrics.f1_score(
                yt, y, labels=self.get_label_indices(), average="macro", zero_division=0
            )
        )

    def weightedf1_score(self, yt, y):
        return self.natural_score(
            metrics.f1_score(
                yt,
                y,
                labels=self.get_label_indices(),
                average="weighted",
                zero_division=0,
            )
        )

    def precision_score(self, yt, y, metric="macro"):
        return self.natural_score(
            metrics.precision_score(
                yt, y, labels=self.get_label_indices(), average=metric, zero_division=0
            )
        )

    def recall_score(self, yt, y, metric="macro"):
        return self.natural_score(
            metrics.recall_score(
                yt, y, labels=self.get_label_indices(), average=metric, zero_division=0
            )
        )

    def eachf1_score(self, yt, y):
        return self.natural_score(
            metrics.f1_score(
                yt, y, labels=self.get_label_indices(), average=None, zero_division=0
            )
        )

    def eachprecision_score(self, yt, y):
        return self.natural_score(
            metrics.precision_score(
                yt, y, labels=self.get_label_indices(), average=None, zero_division=0
            )
        )

    def eachrecall_score(self, yt, y):
        return self.natural_score(
            metrics.recall_score(
                yt, y, labels=self.get_label_indices(), average=None, zero_division=0
            )
        )

    def accuracy(self, yt, y):
        return self.natural_score(metrics.accuracy_score(yt, y))

    def get_metrics(self):
        flatten_golds = self.get_golds()
        flatten_preds = self.get_preds()

        results = dict(
            label_names=[x[0] for x in self.get_labels()],
            eachprecision=self.eachprecision_score(yt=flatten_golds, y=flatten_preds),
            eachrecall=self.eachrecall_score(yt=flatten_golds, y=flatten_preds),
            eachf1=self.eachf1_score(yt=flatten_golds, y=flatten_preds),
            macroprecision=self.precision_score(
                yt=flatten_golds, y=flatten_preds, metric="macro"
            ),
            macrorecall=self.recall_score(
                yt=flatten_golds, y=flatten_preds, metric="macro"
            ),
            macrof1=self.macrof1_score(yt=flatten_golds, y=flatten_preds),
            weightedprecision=self.precision_score(
                yt=flatten_golds, y=flatten_preds, metric="weighted"
            ),
            weightedrecall=self.recall_score(
                yt=flatten_golds, y=flatten_preds, metric="weighted"
            ),
            weightedf1=self.weightedf1_score(yt=flatten_golds, y=flatten_preds),
            accuracy=self.accuracy(yt=flatten_golds, y=flatten_preds),
        )

        return results
