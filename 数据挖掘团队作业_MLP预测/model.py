import torch
import torch.nn as nn



class MLP(nn.Module):
        def __init__(self, input_size, common_size):
            super(MLP, self).__init__()
            self.linear = nn.Sequential(
                nn.Linear(input_size, input_size // 2),
                nn.ReLU(inplace=True),
                nn.Linear(input_size // 2, input_size // 4),
                nn.ReLU(inplace=True),
                nn.Linear(input_size // 4, common_size)
            )
 
        def forward(self, x):
            x = torch.from_numpy(x)
            x = x.float()
            out = self.linear(x)
            return out
