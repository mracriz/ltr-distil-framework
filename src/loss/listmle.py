"""Optional ListMLE loss.

Use this only if you have a full teacher order, not a single winner.
SetwiseTrainer does not call it. The heapsort trace in SetwiseTeacher.rank
gives winners; a full order is the best-first list returned by rank.

teacher_order is a long tensor [batch, k] of indices into the set, best first.
Under the Plackett-Luce model the loss is the negative log-likelihood of
that permutation:

    for position i in the permutation (starting at the best document):
        loss += -log softmax(scores of the documents not yet placed)_i

Average over the batch.
"""

import torch


def listmle_loss(scores: torch.Tensor, teacher_order: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(
        "Implement listmle_loss. This is optional. The main path is setwise_loss."
    )
