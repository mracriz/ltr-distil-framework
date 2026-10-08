"""Pairwise distillation loss.

The teacher compared two documents and picked a winner. The student has
already scored each document on its own. For winner y and the other
document j, RankNet penalizes the student when s_j is not below s_y:

    softplus(-(s_y - s_j)) = log(1 + exp(-(s_y - s_j)))

Average that over every other document in the row, then over the batch.
A pair is a row of width 2, so each row has one term. A higher score must
mean "more relevant", because inference sorts by this same number.

Check yourself on scores [[2.0, 0.0]] and winner 0.
The loss is about 0.1269.
"""

import torch


def pairwise_ranknet_loss(
    scores: torch.Tensor,
    winner_index: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError(
        "Implement pairwise_ranknet_loss. "
        "scores [[2.0, 0.0]] and winner 0 should give a value near 0.1269."
    )
