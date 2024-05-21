import logging

import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.modules.loss import _Loss






class CustomPLModule(pl.LightningModule):
    """
    This Module ensures that not only the target tensor but also the input tensors
    of the training batch are passed into the loss function
    """

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """performs the training step"""
        output = self._produce_train_output(train_batch[:-1])
        target = train_batch[-1]  # By convention target is always the last element returned by datasets,
        # but we skip this step here and move it to the loss function
        # in order to retrieve context-related data inside the loss function

        # # saves tensors to a file for further inspection
        # for i, x in enumerate(train_batch[:-1]):
        #     logging.info(x.shape)
        #     for j in range(x.shape[-1]):
        #         slice_tensor = x[:, :, j]
        #         logging.info(f"{i}, {j}")
        #         pd.DataFrame(slice_tensor.numpy()).to_csv(f"debug/{batch_idx}_encoded_batch_0{i}_tensor_0{j}_slice.csv", index=False)  # save to file
        loss = self._compute_loss(output, train_batch)
        self.log(
            "train_loss",
            loss,
            batch_size=train_batch[0].shape[0],
            prog_bar=True,
            sync_dist=True,
        )
        self._calculate_metrics(output, target, self.train_metrics)
        return loss


class CustomLoss(nn.Module):
    def __init__(self):
        logging.debug('Initializing custom loss')
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        real_target = target[-1]  # last element is the target element
        logging.debug(f'Calculating loss, having these shapsies, {output.shape}, {real_target.shape}')
        loss = torch.mean((output - real_target) ** 2)
        return loss
