import logging
import os
from typing import Tuple, Any

import numpy as np
import pandas as pd
from darts.logging import raise_if_not
from darts.models import RNNModel
from darts.models.forecasting.rnn_model import _RNNModule
from darts.models.forecasting.tft_model import _TFTModule, TFTModel, MixedCovariatesTrainTensorType, logger
from darts.models.forecasting.tft_submodels import get_embedding_size
from darts.utils import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import torch

from neuro_symbolic_demand_forecasting.darts.loss import CustomPLModule


# For LSTM
class ExtendedRNNModel(RNNModel):

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        kwargs = {}
        if isinstance(self.rnn_type_or_module, str):
            kwargs["name"] = self.rnn_type_or_module

        logging.debug("Initiating custom rnn module")
        return _CustomRNNModule(
            input_size=input_dim,
            target_size=output_dim,
            nr_params=nr_params,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            num_layers=self.n_rnn_layers,
            **self.pl_module_params,
            **kwargs,
        )


class _CustomRNNModule(_RNNModule, CustomPLModule):
    def training_step(self, train_batch, batch_idx):
        # Call the training_step method from _CustomPLModule
        logging.debug("Delegating training_step")
        return CustomPLModule.training_step(self, train_batch, batch_idx)


# For TFT
class ExtendedTFTModel(TFTModel):

    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        # add custom TFTModule here!
        # module: nn.Module = super()._create_model(train_sample)
        # print(module.pl_module_params)
        (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariates,
            future_target,
        ) = train_sample

        # add a covariate placeholder so that relative index will be included
        if self.add_relative_index:
            time_steps = self.input_chunk_length + self.output_chunk_length

            expand_future_covariate = np.arange(time_steps).reshape((time_steps, 1))

            historic_future_covariate = np.concatenate(
                [
                    ts[: self.input_chunk_length]
                    for ts in [historic_future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
            future_covariate = np.concatenate(
                [
                    ts[-self.output_chunk_length:]
                    for ts in [future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )

        self.output_dim = (
            (future_target.shape[1], 1)
            if self.likelihood is None
            else (future_target.shape[1], self.likelihood.num_parameters)
        )

        tensors = [
            past_target,
            past_covariate,
            historic_future_covariate,  # for time varying encoders
            future_covariate,
            future_target,  # for time varying decoders
            static_covariates,  # for static encoder
        ]
        type_names = [
            "past_target",
            "past_covariate",
            "historic_future_covariate",
            "future_covariate",
            "future_target",
            "static_covariate",
        ]
        variable_names = [
            "target",
            "past_covariate",
            "future_covariate",
            "future_covariate",
            "target",
            "static_covariate",
        ]

        variables_meta = {
            "input": {
                type_name: [f"{var_name}_{i}" for i in range(tensor.shape[1])]
                for type_name, var_name, tensor in zip(
                    type_names, variable_names, tensors
                )
                if tensor is not None
            },
            "model_config": {},
        }

        reals_input = []
        categorical_input = []
        time_varying_encoder_input = []
        time_varying_decoder_input = []
        static_input = []
        static_input_numeric = []
        static_input_categorical = []
        categorical_embedding_sizes = {}
        for input_var in type_names:
            if input_var in variables_meta["input"]:
                vars_meta = variables_meta["input"][input_var]
                if input_var in [
                    "past_target",
                    "past_covariate",
                    "historic_future_covariate",
                ]:
                    time_varying_encoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["future_covariate"]:
                    time_varying_decoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["static_covariate"]:
                    if (
                            self.static_covariates is None
                    ):  # when training with fit_from_dataset
                        static_cols = pd.Index(
                            [i for i in range(static_covariates.shape[1])]
                        )
                    else:
                        static_cols = self.static_covariates.columns
                    numeric_mask = ~static_cols.isin(self.categorical_embedding_sizes)
                    for idx, (static_var, col_name, is_numeric) in enumerate(
                            zip(vars_meta, static_cols, numeric_mask)
                    ):
                        static_input.append(static_var)
                        if is_numeric:
                            static_input_numeric.append(static_var)
                            reals_input.append(static_var)
                        else:
                            # get embedding sizes for each categorical variable
                            embedding = self.categorical_embedding_sizes[col_name]
                            raise_if_not(
                                isinstance(embedding, (int, tuple)),
                                "Dict values of `categorical_embedding_sizes` must either be integers or tuples. Read "
                                "the TFTModel documentation for more information.",
                                logger,
                            )
                            if isinstance(embedding, int):
                                embedding = (embedding, get_embedding_size(n=embedding))
                            categorical_embedding_sizes[vars_meta[idx]] = embedding

                            static_input_categorical.append(static_var)
                            categorical_input.append(static_var)

        variables_meta["model_config"]["reals_input"] = list(dict.fromkeys(reals_input))
        variables_meta["model_config"]["categorical_input"] = list(
            dict.fromkeys(categorical_input)
        )
        variables_meta["model_config"]["time_varying_encoder_input"] = list(
            dict.fromkeys(time_varying_encoder_input)
        )
        variables_meta["model_config"]["time_varying_decoder_input"] = list(
            dict.fromkeys(time_varying_decoder_input)
        )
        variables_meta["model_config"]["static_input"] = list(
            dict.fromkeys(static_input)
        )
        variables_meta["model_config"]["static_input_numeric"] = list(
            dict.fromkeys(static_input_numeric)
        )
        variables_meta["model_config"]["static_input_categorical"] = list(
            dict.fromkeys(static_input_categorical)
        )

        n_static_components = (
            len(static_covariates) if static_covariates is not None else 0
        )

        self.categorical_embedding_sizes = categorical_embedding_sizes
        return _CustomTFTModule(self.output_dim, variables_meta, n_static_components,
                                self.hidden_size,
                                self.lstm_layers, self.num_attention_heads, self.full_attention,
                                self.feed_forward,
                                self.hidden_continuous_size, self.categorical_embedding_sizes, self.dropout,
                                self.add_relative_index, self.norm_type, **self.pl_module_params)


class _CustomTFTModule(_TFTModule, CustomPLModule):
    # train_losses = []
    # val_losses = []
    # save_to_path = os.environ["MODEL_PATH"]
    #
    # def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
    #     if batch_idx % 50 == 0:
    #         loss = outputs['loss'].item()
    #         self.train_losses.append(loss)
    #         logging.debug(f'Batch {batch_idx}, Training Loss: {loss}')
    #
    # def on_validation_batch_end(
    #         self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    # ) -> None:
    #     logging.info(f"Validationoutput {outputs}")
    #     if batch_idx % 50 == 0 and batch_idx > 0:
    #         loss = outputs['loss'].item()
    #         self.val_losses.append(loss)
    #         logging.debug(f'Batch {batch_idx}, Validation Loss: {loss}')
    #
    # def on_fit_end(self) -> None:
    #     import csv
    #     with open("/filepath",'wb') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['loss', 'age'])
    #         writer.writerow(['John Doe', 30])

    def training_step(self, train_batch, batch_idx):
        # Call the training_step method from _CustomPLModule
        logging.debug("Delegating training_step")
        return CustomPLModule.training_step(self, train_batch, batch_idx)


class EarlyStoppingAfterNthEpoch(EarlyStopping):
    def __init__(self, monitor: str, min_delta: float, patience: int, verbose: bool, start_epoch: int):
        self.start_epoch = start_epoch
        super(EarlyStoppingAfterNthEpoch, self).__init__(monitor, min_delta, patience, verbose)

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            self._run_early_stopping_check(trainer)
        else:
            self.best_score = torch.tensor(1)  # just setting something arbitrary high
            logging.info(
                f"Skipping epoch {trainer.current_epoch} for the early stopping check, starting after {self.start_epoch} >:D")

    # def on_train_end(self, trainer, pl_module):
    #     # instead, do it at the end of training loop
    #     # self._run_early_stopping_check(trainer)


class LossCurveCallback(Callback):
    """
    Callback to store the loss values to a file for further evaluation
    """

    def __init__(self, folder: str, nth_batch: int = 1):
        self.folder = folder
        self.nth_batch = nth_batch

    train_losses = {}
    val_losses = {}

    # def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
    #     if batch_idx % 50 == 0:
    #         loss = outputs['loss'].item()
    #         self.train_losses.append(loss)
    #         logging.debug(f'Batch {batch_idx}, Training Loss: {loss}')
    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
            batch_idx: int
    ) -> None:
        if batch_idx % self.nth_batch == 0:
            loss = outputs['loss'].item()
            self.train_losses[f'{trainer.current_epoch}.{batch_idx}'] = loss
            logging.debug(f'Batch {batch_idx}, Training Loss: {loss}')

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if batch_idx % self.nth_batch == 0:
            loss = outputs.item()
            self.val_losses[trainer.current_epoch] = loss
            logging.debug(f'Batch {batch_idx}, Validation Loss: {loss}')

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        import csv
        with open(f"{self.folder}/train_losses.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['batch', 'loss'])
            rows = [[i, j] for i, j in self.train_losses.items()]
            writer.writerows(rows)

        with open(f"{self.folder}/val_losses.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['batch', 'loss'])
            rows = [[i, j] for i, j in self.val_losses.items()]
            writer.writerows(rows)
