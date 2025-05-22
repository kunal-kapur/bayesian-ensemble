'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

from model import ConsistentMCDropout
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_prob=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut_active = False
        self.shortcut_lauyer = None
        self.dropout_shortcut = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_active = True
            self.shortcut_layer = nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            self.dropout_shortcut = ConsistentMCDropout(dropout_prob)
        self.dropout1 = ConsistentMCDropout(dropout_prob)
        self.dropout2 = ConsistentMCDropout(dropout_prob)

        self.layer_importance_dict = {"conv1": {}, "conv2": {}, "shortcut": {}}



    def forward(self, x, mask):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.dropout1(out, mask=mask)
        if self.shortcut_active:
            shortcut_out = self.dropout_shortcut(self.shortcut_layer(x), mask=mask)
            out = shortcut_out + out
        else:
            out = x + out
        out = F.relu(out)
        out = self.dropout2(out, mask=mask)
        return out

    def propagate_importance_score(self, prev_scores, prev_weights, mask):
        # Importance from conv2 and dropout2
        conv2_input_scores = torch.zeros_like(prev_scores[0])
        for i in range(len(prev_scores)):
            dropout2_mask = self.dropout2.mask_dict[mask].squeeze(dim=0)
            conv2_weights = self.conv2.weight

            out_channels, in_channels, kH, kW = prev_weights[i].shape
            masked_scores = (prev_scores[i] * dropout2_mask).squeeze()
            conv2_input_scores_flat = torch.matmul(torch.abs(prev_weights[i].T), masked_scores)
            conv2_input_scores = conv2_input_scores_flat.view(in_channels, 1, 1)

            if self.shortcut_active:
                shortcut_mask = self.dropout_shortcut.mask_dict[mask].squeeze(dim=0)
                shortcut_weights = self.shortcut_layer.weight
                out_channels, in_channels, kH, kW = shortcut_weights.shape
                masked_scores = (prev_scores[i] * shortcut_mask).squeeze()
                shortcut_input_scores_flat = torch.matmul(torch.abs(shortcut_weights.T), masked_scores)
                shortcut_input_scores = shortcut_input_scores_flat.view(in_channels, 1, 1)

        dropout1_mask = self.dropout1.mask_dict[mask].squeeze(dim=0)
        out_channels, in_channels, kH, kW = conv2_weights.shape
        masked_scores = (conv2_input_scores * dropout1_mask).squeeze()
        conv1_input_scores_flat = torch.matmul(torch.abs(conv2_weights.T), masked_scores)
        conv1_input_scores = conv1_input_scores_flat.view(in_channels, 1, 1)

        self.layer_importance_dict['conv2_input_scores'][mask] = (conv2_input_scores)
        self.layer_importance_dict['conv1_input_scores'][mask] = (conv1_input_scores)
        self.layer_importance_dict['shortcut_input_scores'][mask] = (shortcut_input_scores)

        return [conv1_input_scores, shortcut_input_scores], [self.conv1.weight, self.shortcut_layer.weight]
    
    def prune(self, threshold):
        aggregated_scores = {}
        for layer, scores_dict in self.layer_importance_dict.items():
            for mask, scores in scores_dict.items():
                if layer not in aggregated_scores:
                    aggregated_scores[layer] = scores.clone()
                else:
                    aggregated_scores[layer] += scores
            aggregated_scores[layer] /= len(scores_dict)

        for layer, scores in aggregated_scores.items():
            mask = scores < threshold
            if layer == "conv1":
                self.conv1.weight.data = prune.CustomFromMask.apply(self.conv1.weight.data, mask)
            elif layer == "conv2":
                self.conv2.weight.data = prune.CustomFromMask.apply(self.conv2.weight.data, mask)
            elif layer == "shortcut" and self.shortcut_active:
                self.shortcut_layer.weight.data = prune.CustomFromMask.apply(self.shortcut_layer.weight.data, mask)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()