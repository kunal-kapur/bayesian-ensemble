from torch import nn
import torch
import torch.nn.functional as F
import warnings


class Net(nn.Module):
    def __init__(self, num_masks, dropout_probs=[0.8]):
        super(Net, self).__init__()

        self.num_masks = num_masks
        layer_sizes = [784, 256, 10]
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.dropout1 = ConsistentMCDropout(p=dropout_probs[0])
        self.fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        

    def forward(self, x, mask):
        mask = max(0, mask)
        mask = min(self.num_masks - 1, mask)

        if mask > self.num_masks:
            warnings.warn("The provided mask is below 0. clipping to 0", UserWarning)

        elif mask > self.num_masks:
            warnings.warn(f"The provided mask exceeds the total number of masks added, clipping to {self.num_masks - 1}", UserWarning)
        x  = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x, mask)
        x = self.fc2(x)
        output_logits = torch.log_softmax(x, dim=1) # compute numerically stable softmax for fitting
        return output_logits
    

class _ConsistentMCDropoutMask(nn.Module):
    """Taken and modified from here: https://blackhc.github.io/batchbald_redux/consistent_mc_dropout.html
    """
    def __init__(self, p=0.8):
        super().__init__()

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))

        self.p = p
        self.mask_dict = {}

    def extra_repr(self):
        return "p={}".format(self.p)

    def reset_mask(self):
        self.mask = None

    def _create_mask(self, input, num):
        mask_shape = list((input.shape[1:]))
        mask = torch.empty(mask_shape, dtype=torch.bool, device=input.device).bernoulli_(self.p)
        self.mask_dict[num] = mask
        return mask

    def forward(self, input: torch.Tensor, m: int):
        if self.p == 0:
            return input
        if m not in self.mask_dict:
            self._create_mask(input=input, num=m)
        
        given_mask = self.mask_dict[m]
        mc_output = input.masked_fill(given_mask.unsqueeze(dim=0), 0) / (1 - self.p)
        return mc_output
    


    
class ConsistentMCDropout(_ConsistentMCDropoutMask):
    r"""Randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. The elements to zero are randomized on every forward call during training time.

    During eval time, a fixed mask is picked and kept until `reset_mask()` is called.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: `Any`. Input can be of any shape
        - Output: `Same`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """
    pass


class MCMC:
    def __init__(self, model: Net, increment_amt):
        self.model = model
        self.num_masks = model.num_masks
        self.ocurrences = torch.ones(self.num_masks, 1)
        self.tot = sum(self.ocurrences)
        # self.transition_probs = torch.ones(num_masks, 1)
        self.increment_amt = increment_amt
        self.previous = 1
        self.dist = None
        self.calculated = False

    def recalculate_dist(self):
        self.dist = (self.ocurrences / self.tot).squeeze(dim=1)


    def increment(self, mask):
        self.ocurrences[mask] += self.increment_amt
        self.tot += self.increment_amt

    def calculate_liklihood(self, x, y, mask):
        logits = self.model.forward(x=x, mask=mask)
        val = torch.sum(logits[:, y])
        return val
        
    def transition(self, x, y):
        self.calculated = False
        proposed_mask = torch.randint(0, self.num_masks, (1,))
        ratio = self.calculate_liklihood(x=x, y=y, mask=proposed_mask) / self.calculate_liklihood(x=x, y=y, mask=self.previous)
        acceptance_prob = torch.tensor([min(torch.tensor([1]).float(), ratio)])
        accepted = torch.bernoulli(acceptance_prob).item()
        if (accepted == 1):
            self.previous = proposed_mask
            self.increment(proposed_mask)


    def modify_distribution(self, keep_indices):

        keep_indices = torch.tensor(keep_indices, dtype=torch.long)
        mask = torch.zeros_like(self.dist, dtype=torch.bool)
        mask[keep_indices] = True

        new_dist = self.dist.detach().clone()
        new_dist[~mask] = 0  # Zero out the probabilities of removed indices

        new_dist[mask] /= new_dist[mask].sum()

        return new_dist


        

    def predict(self, x, chosen_masks=None):
        if self.calculated is False:
            self.recalculate_dist()
            self.calculated = True
        chosen_dist = self.dist

        if chosen_masks is not None:
            chosen_dist = self.modify_distribution(chosen_masks)

        predictions = []
        predictions = [self.model.forward(x, mask=(m)) * chosen_dist[m] for m in range(self.num_masks)]
        prob_tensor = torch.stack(predictions, dim=0)
        return torch.sum(prob_tensor, dim=0)
		
