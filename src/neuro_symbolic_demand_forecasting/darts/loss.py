import torch
import torch.nn as nn

# Define your custom loss function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        # Define your loss components here

    def forward(self, output, target):
        # Calculate your custom loss
        # print('outputted', output)
        # print('target', target)
        loss = torch.mean((output - target)**2)  # Example custom loss (MSE)
        return loss
