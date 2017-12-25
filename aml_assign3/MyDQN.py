import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        in_feature=4
        out_feature=10
        action_num=2
        super(MLP, self).__init__()
        self.fc=nn.Linear(in_feature,out_feature)
        self.fc.weight.data.normal_(0, 0.1)
        self.out=nn.Linear(out_feature,action_num)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, *input):
        x=self.fc(input)
        return self.out(x)


mlp=MLP()