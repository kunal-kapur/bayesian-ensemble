from torch import nn
import torch
import torch.nn.functional as F
import warnings

class Net(nn.Module):
    def __init__(self, num_masks):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)



        self.num_masks = num_masks

        layer_sizes = [4608, 128, 10]
        self.dropout1 = ConsistentMCDropout(num_masks=num_masks, input_shape=layer_sizes[1], p=0.5)
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.dropout1 = ConsistentMCDropout(num_masks=num_masks, input_shape=layer_sizes[1], p=0.8)
        self.fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])

        self.fcs = [self.fc1, self.fc2]

        for i in range(num_masks):
            mask = []
            for j in range(1, len(layer_sizes) - 1):
                mask.append(layer_sizes[j])
            self.dropout_masks[i] = mask
        

    def forward(self, x, mask):

        mask = max(1, mask)
        mask = min(self.num_masks, self.num_masks)

        if mask > self.num_masks:
            warnings.warn("The provided mask exceeds the total number of masks added in as a parameter", UserWarning)
        mask = self.dropout_masks[mask]
        x  = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x, mask)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu()
        x = self.dropout2(x, mask)
        x = self.fc2(x, mask=mask)
        output = F.log_softmax(x, dim=1)
        return output
    

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
        mask_shape = list((input.shape))
        mask = torch.empty(mask_shape, dtype=torch.bool, device=input.device).bernoulli_(self.p)
        self.mask_dict[num] = mask
        return mask

    def forward(self, input: torch.Tensor, m: int):
        if self.p == 0.0:
            return input
        
        if m not in self.mask_dict:
            self._create_mask(m)
        
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
	
	def __init__(self, model: Net, num_masks, increment_amt):

		self.model = model
		self.num_masks = num_masks

		self.ocurrences = torch.ones(num_masks, 1)
		self.tot = sum(self.dist)
		self.transition_probs = torch.ones(num_masks, 1)
		self.increment_amt = increment_amt
		self.previous = 1

	def calculate_dist(self):
		self.dist = self.ocurrences / self.tot


	def increment(self, mask):
		self.dist[mask] += self.increment_amt
		self.tot += self.increment_amt
		return self.dist

	def calculate_liklihood(self, x, y, mask):
		out = self.model.forward(x=x, mask=mask)
		return out[y]
		
	def transition(self, x, y):
		proposed_mask = torch.randint(0, self.num_masks, (1,))
		ratio = self.calculate_liklihood(x=x, y=y, m=proposed_mask) / self.calculate_liklihood(x=x, y=y, m=self.previous)
		acceptance_prob = torch.tensor([min(1, ratio * self.transition_prob[proposed_mask] / self.transition_probs[self.previous])])
		accepted = torch.bernoulli(acceptance_prob).item()
		if (accepted == 1):
			self.previous = proposed_mask
			self.increment(proposed_mask)
		
		self.calculate_dist()

	def predict(self, x):
		predictions = []
		predictions = [self.model.forward(x.unsqueeze(0), mask=m).squeeze(dim=0) * self.num_masks for m in range(self.num_masks)]
		prob_tensor = torch.cat(predictions, dim=0)
		return torch.sum(prob_tensor, dim=0)
		
