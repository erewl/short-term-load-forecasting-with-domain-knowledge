import logging

import argparse
import os
import pickle

import torch.nn as nn
import numpy as np
import optuna
import yaml
import datetime as dt
import pytorch_lightning as pl
from darts.metrics import smape
from darts.models import TFTModel
from optuna.samplers import RandomSampler
from optuna.visualization import plot_param_importances, plot_contour, plot_optimization_history
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping

from neuro_symbolic_demand_forecasting.darts.custom_modules import ExtendedTFTModel
from neuro_symbolic_demand_forecasting.darts.loss import CustomLoss
from neuro_symbolic_demand_forecasting.encoders.encoders import create_encoders, TFT_MAPPING, WEIGHTS
from neuro_symbolic_demand_forecasting.main_train import load_csvs, create_timeseries_from_dataframes, \
    get_trainer_kwargs


# workaround to fix the "Expected parent" error: https://github.com/optuna/optuna/issues/4689
class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def print_callback(study, trial):
    logging.info(f"Current value: {trial.value}, Current params: {trial.params}")
    logging.info(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


def main_optimize(smart_meter_files: list[str], weather_forecast_files: list[str], weather_actuals_files: list[str],
                  _full_config: dict, _model_folder: str):
    sm, wf, wa = load_csvs(model_config, smart_meter_files, weather_forecast_files, weather_actuals_files)
    smart_meter_tss, weather_forecast_ts, weather_actuals_ts = create_timeseries_from_dataframes(sm, wf, wa,
                                                                                                 scale=True,
                                                                                                 add_static_covariates=True,
                                                                                                 pickled_scaler_folder=path)

    _config = model_config['tft_config']

    def objective(trial):

        def get_suggestion(suggest_fn, name):
            if _config[name]['increment'] == 0:
                return _config[name]['start']
            else:
                logging.info(f"Parameter {name} will be tuned.")
                return suggest_fn(name,
                                  low=_config[name]['start'],
                                  high=_config[name]['end'],
                                  step=_config[name]['increment'])

        # early_stopper = EarlyStopping("val_loss", min_delta=0.0001, patience=10, verbose=True)
        pruning_callback: list = [OptunaPruning(trial, monitor="val_loss")]

        pl_trainer_kwargs, num_workers = get_trainer_kwargs(_full_config, callbacks=pruning_callback)
        # pl_trainer_kwargs['max_epochs'] = _config['n_epochs']
        trainer = pl.Trainer(max_epochs=_config['n_epochs'], callbacks=pruning_callback)
        # TODO add early stopping here too

        input_chunk_length = get_suggestion(trial.suggest_int, 'input_chunk_length')
        hidden_size = get_suggestion(trial.suggest_int, 'hidden_size')
        hidden_continuous_size = get_suggestion(trial.suggest_int, 'hidden_continuous_size')
        num_attention_heads = get_suggestion(trial.suggest_int, 'num_attention_heads')
        lstm_layers = get_suggestion(trial.suggest_int, 'lstm_layers')
        batch_size = get_suggestion(trial.suggest_int, 'batch_size')
        dropout = get_suggestion(trial.suggest_float, 'dropout')

        use_static_covariates = trial.suggest_categorical("use_static_covariates", [True])
        add_relative_index = trial.suggest_categorical("add_relative_index", [False])

        match _config.get('loss_fn'):
            case 'Custom':
                loss_fn = CustomLoss(TFT_MAPPING, {}, WEIGHTS)
                # model_cls = ExtendedTFTModel
                model = ExtendedTFTModel(
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=_config['output_chunk_length'],
                    hidden_size=hidden_size,
                    hidden_continuous_size=hidden_continuous_size,
                    num_attention_heads=num_attention_heads,
                    lstm_layers=lstm_layers,
                    batch_size=batch_size,
                    add_encoders=create_encoders('TFT'),
                    dropout=dropout,
                    loss_fn=loss_fn,
                    use_static_covariates=use_static_covariates,
                    add_relative_index=add_relative_index,
                    optimizer_kwargs=_config['optimizer_kwargs'],
                    # pl_trainer_kwargs=pl_trainer_kwargs
                )
            case other:
                logging.info(f'{other} not implemented yet, falling back to MSE')
                loss_fn = nn.MSELoss()
                model = TFTModel(
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=_config['output_chunk_length'],
                    hidden_size=hidden_size,
                    hidden_continuous_size=hidden_continuous_size,
                    num_attention_heads=num_attention_heads,
                    lstm_layers=lstm_layers,
                    batch_size=batch_size,
                    add_encoders=create_encoders('TFT'),
                    dropout=dropout,
                    loss_fn=loss_fn,
                    use_static_covariates=use_static_covariates,
                    add_relative_index=add_relative_index,
                    optimizer_kwargs=_config['optimizer_kwargs'],
                    # pl_trainer_kwargs=pl_trainer_kwargs
                )

        input_chunk_length = model.input_chunk_length
        output_chunk_length = model.output_chunk_length
        validation_set_size = input_chunk_length + output_chunk_length
        train_tss, val_tss = zip(*[sm.split_after(len(sm) - validation_set_size - 1) for sm in smart_meter_tss])
        train_tss = list(train_tss)
        val_tss = list(val_tss)

        model.fit(
            series=train_tss,
            past_covariates=weather_actuals_ts,
            future_covariates=weather_forecast_ts,
            val_series=smart_meter_tss,
            val_past_covariates=weather_actuals_ts,
            val_future_covariates=weather_forecast_ts,
            epochs=_config['n_epochs'],
            trainer=trainer,
            num_loader_workers=num_workers,
        )

        # Evaluate how good it is on the validation set
        preds = model.predict(series=train_tss,
                              past_covariates=weather_actuals_ts,
                              future_covariates=weather_forecast_ts, n=output_chunk_length)

        smapes = smape(val_tss, preds, n_jobs=-1, verbose=True)
        smape_val = np.mean(smapes)

        return smape_val if smape_val != np.nan else float("inf")

    study = optuna.create_study(study_name='test', direction="minimize", sampler=RandomSampler())
    # sadly we can only run one at a time?????
    study.optimize(objective, n_jobs=1, n_trials=3, callbacks=[print_callback])

    logging.info(f"Best params: {study.best_params}")
    logging.info(f"Best value: {study.best_value}")
    logging.info(f"Best Trial: {study.best_trial}")
    logging.info(f"Trials: {study.trials}")

    with open(f'{_model_folder}/study.pkl', 'wb') as f:
        pickle.dump(study, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune a model based on hyperparameters ranges in config file')
    parser.add_argument('-smd', '--smart-meter-data', metavar='SMD_FILES', type=str, nargs='+',
                        help='comma-separated list of smart meter data csv files to be used for training')
    parser.add_argument('-wfd', '--weather-forecast-data', metavar='WFD_FILES', type=str, nargs='+',
                        help='comma-separated list of weather forecast csv files to be used for training')
    parser.add_argument('-wad', '--weather-actuals-data', metavar='WAD_FILES', type=str, nargs='+',
                        help='comma-separated list of weather actuals csv files to be used for training')
    parser.add_argument('-md', '--model-configuration', metavar='MODEL_CONFIG_PATH', type=str,
                        help='path to the model configuration YAML file')
    parser.add_argument('-sv', '--save-model-as', metavar='MODEL_SAVE_PATH', type=str,
                        help='path where model should be saved')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    with open(args.model_configuration, 'r') as file:
        logging.info(f'Loading config from {args.model_configuration}')
        model_config = yaml.safe_load(file)

    # Creating folder to save scalers and model 'YYYYMMDD_HHMM'
    current_datetime = dt.datetime.now().strftime('%Y%m%d_%H%M')
    base_dir, last_folder = os.path.split(args.save_model_as)
    new_last_folder = f"{current_datetime}_{last_folder}"
    path = os.path.join(base_dir, new_last_folder)
    os.makedirs(path)
    logging.info(f"Saving everything related to this model training run at: {path}")

    smd_files, wfd_files, wad_files = [], [], []
    if args.smart_meter_data:
        smd_files = [file.strip() for file in ','.join(args.smart_meter_data).split(',')]
    if args.weather_forecast_data:
        wfd_files = [file.strip() for file in ','.join(args.weather_forecast_data).split(',')]
    if args.weather_actuals_data:
        wad_files = [file.strip() for file in ','.join(args.weather_actuals_data).split(',')]

    main_optimize(smd_files, wfd_files, wad_files, model_config, path)
