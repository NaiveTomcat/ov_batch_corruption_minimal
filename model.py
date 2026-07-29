import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Minimal model for testing batch corruption in OpenVINO.
    Execution graph:
        input -> split to 2 branches -> each branch has a 1x1 Conv2d
              -> branches are concatenated -> output
    """

    def __init__(self, channels: int = 8, seed: int = 42):
        assert channels % 2 == 0, "Channels must be even for splitting."
        super().__init__()
        self.channels = channels

        torch.manual_seed(seed)

        self.split1 = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.split2 = nn.Conv2d(channels, channels // 2, kernel_size=1)

    def forward(self, x):
        # Split the input tensor into two branches
        branch1 = self.split1(x)
        branch2 = self.split2(x)

        # Concatenate the branches along the channel dimension
        output = torch.cat((branch1, branch2), dim=1)
        return output

def build(channels: int = 8, seed: int = 42) -> nn.Module:
    """
    Build the model with specified channels and seed.
    """
    model = Model(channels=channels, seed=seed)
    model.eval()
    return model

