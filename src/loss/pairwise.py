"""Optional pairwise loss built from a setwise winner.

For winner y and every other document j in the set, penalize the student
when s_j is not below s_y. One common form is softplus(-(s_y - s_j)),
averaged over all those pairs and over the batch.

SetwiseTrainer does not call this. It is here if you want to compare a
pairwise objective against the setwise one on the same labels.
"""

import torch


def pairwise_ranknet_loss(
    scores: torch.Tensor,
    winner_index: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError(
        "Implement pairwise_ranknet_loss. This is optional. "
        "The main path is setwise_loss."
    )
