import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA


class TripletLossFn(nn.Module):
    def __init__(self, eps=1):
        super().__init__()

        self.eps = eps

    def forward(self, a, p, n):
        x = LA.vector_norm(a - p, ord=2, dim=1) - LA.vector_norm(a - n, ord=2, dim=1) + self.eps
        x = F.relu(x)
        return x


CLS_LOSS_FN = nn.CrossEntropyLoss()
REG_LOSS_FN = nn.MSELoss()
TRIP_LOSS_FN = TripletLossFn()
