"""Setwise distillation loss.

The teacher looked at k documents and picked one winner. The student has
already scored those documents independently. This loss treats the k scores
as logits of a single classification: the class is the teacher's winner.

For a set with scores s_1 ... s_k and winner y:

    P(i) = exp(s_i) / sum_j exp(s_j)
    loss = -log P(y)

Average that over the batch. A higher score must mean "more relevant",
because inference sorts by this same number and does not call this loss.

Check yourself on scores [[2.0, 0.0, 0.0]] and winner 0.
The loss is about 0.2395.
"""

import torch
import torch.nn.functional as F


def setwise_loss(scores: torch.Tensor, winner_index: torch.Tensor) -> torch.Tensor:
    """
    Args:
        scores: float tensor [batch, k].
        winner_index: long tensor [batch], index of the teacher winner.

    Returns:
        A scalar, the mean over the batch.
    """
    
    return F.cross_entropy(scores, winner_index)
