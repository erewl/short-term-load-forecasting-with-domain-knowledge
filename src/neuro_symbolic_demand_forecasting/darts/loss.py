import torch
import torch.nn as nn
import pytorch_lightning as pl


class CustomPLModule(pl.LightningModule):
    """
    This Module ensures that not only the target tensor but also the input tensors
    of the training batch are passed into the loss function
    """

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        output = self._produce_train_output(train_batch[:-1])
        loss = self._compute_loss(output, train_batch)
        self.log(
            "train_loss",
            loss,
            batch_size=train_batch[0].shape[0],
            prog_bar=True,
            sync_dist=True,
        )
        # By convention target is always the last element returned by datasets
        self._calculate_metrics(output, train_batch[-1], self.train_metrics)
        return loss


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()


    def forward(self, output, target):
        real_target = target[-1]  # last element is the target element
        loss = torch.mean((output - real_target) ** 2)  # Example custom loss (MSE)
        # loss = torch.sum((output - 500) ** 2)
        # loss = torch.sum(output**2)
        return loss
