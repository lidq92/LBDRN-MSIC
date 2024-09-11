import torch
from ignite.metrics.metric import Metric


class LBDRNPerformance(Metric):
    def __init__(self):
        super(LBDRNPerformance, self).__init__()

    def reset(self):
        self._y_pred = None
        self._y      = None

    def update(self, output):
        y_pred, y = output
        self._y_pred = y_pred if self._y_pred is None else torch.cat((y_pred, self._y_pred), dim=0)
        self._y = y if self._y is None else torch.cat((y, self._y), dim=0)
        
    def compute(self):
        mse = torch.nn.functional.mse_loss(self._y_pred, self._y).item()
        
        return {'MSE': mse}
