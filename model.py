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
        # x = self.dropout1(x, mask)
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

    def _create_mask(self, input: torch.Tensor, num: int):
        """Create and store a dropout mask for a specific identifier (num)."""
        mask_shape = input.shape[1:]  # Exclude batch dimension
        mask = (torch.rand(mask_shape, device=input.device) < self.p)  # Boolean mask
        self.mask_dict[num] = mask

    def forward(self, input: torch.Tensor, m: int):
        if self.p == 0:
            return input
        
        if m not in self.mask_dict:
            self._create_mask(input, m)
        
        given_mask = self.mask_dict[m]
        return input * (~given_mask).unsqueeze(0) / (1 - self.p)  # Invert mask and apply scaling

    
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
        

    def predict(self, x):
        if self.calculated is False:
            self.recalculate_dist()
            self.calculated = True
        predictions = []
        predictions = [self.model.forward(x, mask=(m)) * self.dist[m] for m in range(self.num_masks)]
        prob_tensor = torch.stack(predictions, dim=0)
        return torch.sum(prob_tensor, dim=0)
		
