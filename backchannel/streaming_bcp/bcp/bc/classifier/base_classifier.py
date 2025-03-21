from typing import List

import torch
import torch.nn as nn

from bcp.bc.classifier.abs_classifier import AbsClassifier


class BCClassifier(AbsClassifier):

    def __init__(
        self,
        input_size: int,
        output_size: int = 2,
        dropout_rate: float = 0.1,
        activation=torch.nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.pooling = "mean"

        self.dense = nn.Linear(input_size, input_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation_fn = activation
        self._output_size = output_size

        self.classifier = nn.Linear(input_size, output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        block_masks: torch.Tensor = None,
        **kwargs,
    ):
        if self.pooling == "mean":
            # mean pooling of block
            block_len = torch.sum(block_masks, dim=-1)

            merge_xs_pad = torch.bmm(block_masks, xs_pad)
            expanded_block_len = block_len.unsqueeze(-1).expand_as(merge_xs_pad)
            expanded_block_len = torch.clamp(expanded_block_len, min=1e-9)
            x = merge_xs_pad / expanded_block_len

        elif self.pooling == "max":
            # max pooling of block
            expanded_xs_pad = xs_pad.unsqueeze(1).expand(
                (
                    xs_pad.size(0),
                    block_masks.size(1),
                    xs_pad.size(1),
                    xs_pad.size(2),
                )
            )
            expanded_block_masks = block_masks.unsqueeze(-1).expand_as(expanded_xs_pad)

            x = expanded_xs_pad * expanded_block_masks
            x = torch.max(x, 2)[0]

        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        return logits
