"""PAW v3.0 models: the SmallCNN image classifier.

One model class, fresh per task. Conv 1->32 + BN/ReLU/Pool ->
Conv 32->64 + BN/ReLU/Pool -> Linear 1024->2. 20,962 parameters
for a binary task. features() is the penultimate 1024-dim
vector, used by centroid routing and health diagnostics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 4 * 4, n_classes)

    def conv2_relu(self, x):
        """Post-conv2 ReLU activations, before pooling.

        Used by health diagnostics (dead-channel fraction).
        """
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        return F.relu(self.bn2(self.conv2(x)))

    def features(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        return x.flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))


def param_count(n_classes=2):
    return sum(p.numel() for p in SmallCNN(n_classes).parameters())
