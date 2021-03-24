""" TODO """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
from lightly.loss.memory_bank import MemoryBankModule


class CO2Regularizer(MemoryBankModule):
    """TODO

    """

    def __init__(self,
                alpha: float = 1,
                t_consistency: float = 0.05,
                memory_bank_size: int = 0):

        super(CO2Regularizer, self).__init__(size=memory_bank_size)
        self.kl_div = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')
        self.t_consistency = t_consistency
        self.alpha = alpha

    def _get_pseudo_labels(self,
                           out0: torch.Tensor,
                           out1: torch.Tensor,
                           negatives: torch.Tensor = None):
        """TODO

        """
        batch_size, _ = out0.shape
        if negatives is None:
            # use second batch as negative samples
            l_pos = torch.einsum('nc,nc->n', [out0, out1]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [out0, out1.t()])
            # remove elements on the diagonal
            l_neg = l_neg.masked_select(
                ~torch.eye(batch_size, dtype=bool, device=l_neg.device)
            ).view(batch_size, batch_size - 1)
        else:
            # use memory bank as negative samples
            negatives = negatives.to(out0.device)
            l_pos = torch.einsum('nc,nc->n', [out0, out1]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [out0, negatives.clone().detach()])
            
        # concatenate such that positive samples are at index 0
        logits = torch.cat([l_pos, l_neg], dim=1)
        # divide by temperature
        logits = logits / self.t_consistency
        # calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs


    def forward(self,
                out0: torch.Tensor,
                out1: torch.Tensor):
        """TODO

        """

        # normalize the output to length 1
        out0 = torch.nn.functional.normalize(out0, dim=1)
        out1 = torch.nn.functional.normalize(out1, dim=1)

        # ask memory bank for negative samples and extend it with out1 if 
        # out1 requires a gradient, otherwise keep the same vectors in the 
        # memory bank (this allows for keeping the memory bank constant e.g.
        # for evaluating the loss on the test set)
        out1, negatives = \
            super(CO2Regularizer, self).forward(out1, update=True)
        
        # get log probabilities
        p = self._get_pseudo_labels(out0, out1, negatives)
        q = self._get_pseudo_labels(out1, out0, negatives)
        
        # calculate kullback leibler divergence from log probabilities
        return self.alpha * 0.5 * (self.kl_div(p, q) + self.kl_div(q, p))
