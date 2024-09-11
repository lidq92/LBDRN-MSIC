import torch.nn as nn


class LBDRNLoss(nn.Module):
    def __init__(self):
        super(LBDRNLoss, self).__init__()
        
    def forward(self, y_pred, y):
        loss = nn.functional.mse_loss(y_pred, y)
        
        return loss
