from torch import nn
import torch
import torch.nn.functional as F
import warnings


class NetNormalDropout(nn.Module):
    def __init__(self):
        super(NetNormalDropout, self).__init__()
        layer_sizes = [784, 256, 10]
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.dropout1 = nn.Dropout1d(p=0.2)
        self.fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        

    def forward(self, x):

        x  = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        output_logits = torch.log_softmax(x, dim=1) # compute numerically stable softmax for fitting
        return output_logits
    
class NetMCDropout(nn.Module):
    def __init__(self, num_samples):
        super(NetMCDropout, self).__init__()

        layer_sizes = [784, 256, 10]
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.num_samples = num_samples
        

    def forward(self, x):
        x  = self.fc1(x)
        x = F.relu(x)
        x = nn.Dropout1d(p=0.2)(x)
        x = self.fc2(x)
        output_logits = torch.log_softmax(x, dim=1) # compute numerically stable softmax for fitting
        return output_logits
    
    def mc_dropout_predict(self, x):
        y_preds = []
        for i in range(self.num_samples):
            y_preds.append(self.forward(x))
        # print(torch.stack(y_preds, dim=0).shape)
        # print(torch.mean(torch.stack(y_preds, dim=0), dim=0).shape)
        return torch.mean(torch.stack(y_preds, dim=0), dim=0)

    
    
