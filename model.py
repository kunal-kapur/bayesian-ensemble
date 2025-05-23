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


        self.pool_indices = {}

        self.num_masks = num_masks
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.dropout1 = ConsistentMCDropout(p=dropout_probs[0])
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = ConsistentMCDropout(p=dropout_probs[1])
        self.fc2 = nn.Linear(128, 10)


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
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout1(x, mask)
        # print("prepooling", x.shape)
        x = self.pool(x)
        # print("preflattening", x.shape)
        x = self.flat(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x, mask)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    

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


    def _create_mask(self, input: torch.tensor, num: int) -> torch.Tensor:
        """generates mask and adds to collection of fixed masks

        Args:
            input (torch.tensor): input to be passed into subsequent layers
            num (int): the key (identifier) of the mask to be generated

        Returns:
           torch.tensor: mask for of where 0s will be added for this particular dropout layer
        """
        if input.dim() == 0:
            raise ValueError("Scalar inputs cannot be masked")

        elif input.dim() == 2:
            # For feature dropout (fully connected layers)
            feature_mask = torch.empty(input.shape[1], device=input.device).bernoulli_(1 - self.p)
            mask = feature_mask.view(1, -1)

        elif input.dim() == 4:
            # For channel dropout (CNN layers)
            channel_mask = torch.empty(input.shape[1], device=input.device).bernoulli_(1 - self.p)
            mask = channel_mask.view(1, -1, 1, 1)

        else:
            raise ValueError(f"Unsupported input shape: {input.shape}")

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
        if m not in self.mask_dict:
            self._create_mask(input=input, num=m)
        
        mask = self.mask_dict[m]
        output = (input * mask) / (1 - self.p)
        return output

    


    
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
        self.recalculate_dist()

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
    