from torch import nn
import torch
import torch.nn.functional as F
import warnings
from typing import Union, List


class Net(nn.Module):
    def __init__(self, num_masks: int, dropout_probs: List[int]=[0.8]):
        """Model that incorporates the use of dropout-masks while training

        Args:
            num_masks (_type_): _description_
            dropout_probs (list, optional): _description_. Defaults to [0.8].
        """
        super(Net, self).__init__()

        self.num_masks = num_masks
        layer_sizes = [784, 256, 10]
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.dropout1 = ConsistentMCDropout(p=dropout_probs[0])
        self.fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        

    def forward(self, x: torch.tensor, mask: int)->torch.Tensor:
        """_summary_

        Args:
            x (torch.tensor): (batch_size x 784) input of flattenned MNIST image
            mask (int): the dropout-mask to run the input through

        Returns:
            torch.Tensor: (batch_size x num_classes) tensor  of logits
        """

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
    

class NetV2(nn.Module):
    def __init__(self, num_masks: int, dropout_probs: List[int]=[0.8]):
        """Deeper model that incorporates the use of dropout-masks while training

        Args:
            num_masks (_type_): _description_
            dropout_probs (list, optional): _description_. Defaults to [0.8].
        """
        super(NetV2, self).__init__()

        self.num_masks = num_masks
        layer_sizes = [784, 1024, 512, 10]
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.dropout1 = ConsistentMCDropout(p=dropout_probs[0])
        self.fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.dropout2 = ConsistentMCDropout(p=dropout_probs[1])
        self.fc3 = nn.Linear(layer_sizes[2], layer_sizes[3])


    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state['dropout1_mask_dict'] = self.dropout1.mask_dict
        state['dropout2_mask_dict'] = self.dropout2.mask_dict
        return state
        
    def load_state_dict(self, state_dict, *args, **kwargs):
        dropout1_masks = state_dict.pop('dropout1_mask_dict', {})
        dropout2_masks = state_dict.pop('dropout2_mask_dict', {})
        
        super().load_state_dict(state_dict, *args, **kwargs)
        
        # Restore masks
        self.dropout1.mask_dict = dropout1_masks
        self.dropout2.mask_dict = dropout2_masks

    def forward(self, x: torch.tensor, mask: int)->torch.Tensor:
        """_summary_

        Args:
            x (torch.tensor): (batch_size x 784) input of flattenned MNIST image
            mask (int): the dropout-mask to run the input through

        Returns:
            torch.Tensor: (batch_size x num_classes) tensor  of logits
        """

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
        x = F.relu(x)
        x = self.dropout2(x, mask)
        x = self.fc3(x)
        output_logits = torch.log_softmax(x, dim=1) # compute numerically stable softmax for fitting
        return output_logits
    

class _ConsistentMCDropoutMask(nn.Module):
    """Module to apply apply dropout masks and record new types of dropout masks.
        Taken and modified from here: https://blackhc.github.io/batchbald_redux/consistent_mc_dropout.html
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

    def _create_mask(self, input: torch.tensor, num: int) -> torch.Tensor:
        """generates mask and adds to collection of fixed masks

        Args:
            input (torch.tensor): input to be passed into subsequent layers
            num (int): the key (identifier) of the mask to be generated

        Returns:
           torch.tensor: mask for of where 0s will be added for this particular dropout layer
        """

        mask_shape = list((input.shape[1:]))
        mask = torch.empty(mask_shape, dtype=torch.bool, device=input.device).bernoulli_(self.p)
        self.mask_dict[num] = mask
        return mask

    def forward(self, input: torch.Tensor, m: int) -> torch.Tensor:
        """Runs forward pass through some fixed dropout mask

        Args:
            input (torch.Tensor): input tensor to be passed through dropout layer
            m (int): the mask to be used. Create a new one if it doesn't exist

        Returns:
            torch.Tensor: resulting output of running through dropout layer
        """
        if self.p == 0:
            return input
        if m not in self.mask_dict:
            self._create_mask(input=input, num=m)
        
        given_mask = self.mask_dict[m]
        mc_output = input.masked_fill(given_mask.unsqueeze(dim=0), 0) / (1 - self.p)
        return mc_output
    
    def get_majority_vote_mask(self, threshold=0.5):
        """
        Perform majority voting across all masks in mask_dict.
        
        Args:
            mask_dict: Dictionary of {index: mask_tensor} where mask_tensor is 1 for kept neurons
            threshold: Fraction of masks where neuron must be kept (default: >50%)
        
        Returns:
            Consolidated mask where neurons are kept if they appear in majority of masks
        """
        all_masks = torch.stack(list(self.mask_dict.values()))
        
        keep_counts = all_masks.sum(dim=0)
        
        num_masks = all_masks.shape[0]
        majority_mask = (keep_counts / num_masks) > threshold
        
        return majority_mask.float()
    


    
class ConsistentMCDropout(_ConsistentMCDropoutMask):
    r"""Randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. The elements to zero are randomized on every forward call during training time.


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

        >>> m = ConsistentMCDropout(p=0.8)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)
    """
    pass


class MCMC:
    """Module to apply MCMC learning on the distribution of masks created from our model
    """
    def __init__(self, model: Net, increment_amt):
        self.model = model
        self.num_masks = model.num_masks
        self.ocurrences = torch.ones(self.num_masks, 1)
        self.tot = sum(self.ocurrences)
        self.increment_amt = increment_amt
        self.previous = 1
        self.dist = None
        self.calculated = False

    def recalculate_dist(self):
        """calculates the distriubtion of probabilities of each mask
        """
        self.calculated = True
        self.dist = (self.ocurrences / self.tot).squeeze(dim=1)
        return self.dist


    def increment(self, mask: int):
        """Increase likelihood of a given mask

        Args:
            mask (int): the mask in the distribution to be incremented
        """
        self.ocurrences[mask] += self.increment_amt
        self.tot += self.increment_amt

    def calculate_liklihood(self, x: torch.Tensor, y: torch.Tensor, mask: int) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (batch_size x *input_dim) of input to model
            y (torch.Tensor): (batch_size x 1) of labels
            mask (int): fied mask to be used

        Returns:
            torch.Tensor: sum the log-likihoods from the forward pass on a batch
        """
        logits = self.model.forward(x=x, mask=mask)
        val = torch.sum(logits[:, y])
        return val
        
    def transition(self, x: torch.Tensor, y: torch.Tensor):
        """Running MCMC transition. Using a uniform proposal disitrbutin and changing posterior via lapalace adding

        Args:
            x (torch.Tensor): (batch_size x *input_dim) of input for model
            y (torch.Tensor): (batch_size x 1) of labels
        """
        self.calculated = False
        proposed_mask = torch.randint(0, self.num_masks, (1,))
        ratio = self.calculate_liklihood(x=x, y=y, mask=proposed_mask) / self.calculate_liklihood(x=x, y=y, mask=self.previous)
        acceptance_prob = torch.tensor([min(torch.tensor([1]).float(), ratio)])
        accepted = torch.bernoulli(acceptance_prob).item()
        if (accepted == 1):
            self.previous = proposed_mask
            self.increment(proposed_mask)


    def __modify_distribution(self, keep_indices: torch.Tensor) -> torch.Tensor:
        """Creates a new modified distribution that allows chosen indices to "absorb probability" from other 
        """

        keep_indices = torch.tensor(keep_indices, dtype=torch.long)
        mask = torch.zeros_like(self.dist, dtype=torch.bool)
        mask[keep_indices] = True

        new_dist = self.dist.detach().clone()
        new_dist[~mask] = 0  # Zero out the probabilities of removed indices

        new_dist[mask] /= new_dist[mask].sum()

        return new_dist

    def predict(self, x: torch.Tensor, chosen_masks: torch.Tensor=None) -> torch.Tensor:
        """Uses weighted averaging of each mask to come up with a weighted prediction for the different masks
            instead of randomly taking accepted samples from traditional MCMC in BNNs. This can be done since 
            the space of our discrete masks is small and we can afford to calculate the likelihood of each mask.

        Args:
            x (torch.Tensor): (batch_size x *input_dim) input tensor for model
            chosen_masks (torch.Tensor, optional): tensor that indicates the masks to be kept

        Returns:
            torch.Tensor: weighted prediction of given masks in the distribution
        """
        if self.calculated is False:
            self.recalculate_dist()
        chosen_dist = self.dist

        if chosen_masks is not None:
            chosen_dist = self.__modify_distribution(chosen_masks)

        predictions = []
        predictions = [self.model.forward(x, mask=(m)) * chosen_dist[m] for m in range(self.num_masks)]
        prob_tensor = torch.stack(predictions, dim=0)
        return torch.sum(prob_tensor, dim=0)
		
