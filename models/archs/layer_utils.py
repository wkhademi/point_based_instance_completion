#############################################################
# Some wrappers for PyTorch layers to support channels last #
#############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTranspose1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose1d(*args, **kwargs)

    def forward(self, input):
        output = self.conv_transpose(input.permute(0, 2, 1)).permute(0, 2, 1)

        return output


class ConvTranspose2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(*args, **kwargs)

    def forward(self, input):
        output = self.conv_transpose(input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return output


class BatchNorm2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.batch_norm = nn.BatchNorm2d(*args, **kwargs)

    def forward(self, input):
        norm = self.batch_norm(input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return norm


class Upsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.upsample = nn.Upsample(*args, **kwargs)

    def forward(self, input):
        if len(input.shape) == 3:
            upsampled = self.upsample(input.permute(0, 2, 1)).permute(0, 2, 1)
        elif len(input.shape) == 4:
            upsampled =  self.upsample(input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return upsampled