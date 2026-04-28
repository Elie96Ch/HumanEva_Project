# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:49:18 2026

@author: user
"""

import torch
import torch.nn as nn


class PoseMLP(nn.Module):
    def __init__(self, input_dim=40, output_dim=60, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        layers = []
        in_dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mpjpe_loss(pred, target):
    pred = pred.view(pred.shape[0], -1, 3)
    target = target.view(target.shape[0], -1, 3)
    return torch.norm(pred - target, dim=-1).mean()