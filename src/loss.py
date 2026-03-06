import torch
import torch.nn as nn

class MagnitudeLoss(nn.Module):
    """
    Custom Loss Function unique to the Magnitude Framework.
    Penalizes underestimation of price displacement energy (Magnitude) 
    more heavily than directional errors.
    """
    def __init__(self, energy_weight=2.0):
        super(MagnitudeLoss, self).__init__()
        self.energy_weight = energy_weight
        self.mse = nn.MSELoss()

    def forward(self, prediction, target):
        # Base MSE loss
        base_loss = self.mse(prediction, target)
        
        # Energy Underestimation Penalty: 
        # If the model predicts small movement but a big one happens, we penalize.
        underestimate_mask = (target > prediction).float()
        penalty = underestimate_mask * (target - prediction) ** 2
        
        return base_loss + (self.energy_weight * penalty.mean())

def get_unique_loss():
    return MagnitudeLoss(energy_weight=2.5)
