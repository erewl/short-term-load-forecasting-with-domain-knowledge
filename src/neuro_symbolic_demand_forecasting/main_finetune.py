import logging

import argparse
import os
import torch.nn as nn
import numpy as np
import optuna
import yaml
import datetime as dt
import pytorch_lightning as pl
from darts.metrics import smape
from darts.models import TFTModel
from optuna.visualization import plot_param_importances, plot_contour, plot_optimization_history
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping

from neuro_symbolic_demand_forecasting.darts.custom_modules import ExtendedTFTModel
from neuro_symbolic_demand_forecasting.darts.loss import CustomLoss
from neuro_symbolic_demand_forecasting.main_train import load_csvs, create_timeseries_from_dataframes, \
    create_encoders, get_trainer_kwargs


def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


def create_objective(model_config: dict, data: tuple):
    match model_config['model_class']:
        case 'TFT':
            return create_tft_objective(model_config['tft_config'], model_config, data)
        case default:
            raise Exception(f'No config implement yet for model of type: {default}')


def create_tft_objective(_config: dict, _base_config: dict, data: tuple):
    logging.info(_config)

    sms, wfts, wats = data

    def objective(trial):
        def get_suggestion(suggest_fn, name):
            return suggest_fn(name,
                              low=_config[name]['start'],
                              high=_config[name]['end'],
                              step=_config[name]['increment'])

        pruning_callback: list = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
        pl_trainer_kwargs, num_workers = get_trainer_kwargs(_base_config, callbacks=pruning_callback)

        input_chunk_length = get_suggestion(trial.suggest_int, 'input_chunk_length')
        hidden_size = get_suggestion(trial.suggest_int, 'hidden_size')
        hidden_continuous_size = get_suggestion(trial.suggest_int, 'hidden_continuous_size')
        num_attention_heads = get_suggestion(trial.suggest_int, 'num_attention_heads')
        lstm_layers = get_suggestion(trial.suggest_int, 'lstm_layers')
        batch_size = get_suggestion(trial.suggest_int, 'batch_size')
        dropout = get_suggestion(trial.suggest_float, 'dropout')

        use_static_covariates = trial.suggest_categorical("use_static_covariates", [False, True])
        add_relative_index = trial.suggest_categorical("add_relative_index", [False, True])

        logging.info(
            f"PARAMS: \n input_chunk_length: {input_chunk_length}, "
            f"\n hidden_size: {hidden_size}, "
            f"\n hidden_continuous_size: {hidden_continuous_size}, "
            f"\n num_attention_heads: {num_attention_heads}, "
            f"\n lstm_layers: {lstm_layers}, "
            f"\n batch_size: {batch_size}, "
            f"\n dropout: {dropout}, "
            f"\n use_static_covariates: {use_static_covariates}, "
            f"\n add_relative_index: {add_relative_index}"
        )

        # TODO could be reused in training too
        model_cls = TFTModel
        match _config.get('loss_fn'):
            case 'MSE':
                loss_fn = nn.MSELoss()
            case 'Custom':
                loss_fn = CustomLoss()
                model_cls = ExtendedTFTModel
            case other:
                logging.info(f'{other} not implemented yet, falling back to MSE')
                loss_fn = nn.MSELoss()
        logging.info(f"Initializing Model {model_cls} with {loss_fn}")

        model = model_cls(
            input_chunk_length=input_chunk_length,
            output_chunk_length=_config['output_chunk_length'],
            epochs=100,
            hidden_size=hidden_size,
            hidden_continuous_size=hidden_continuous_size,
            num_attention_heads=num_attention_heads,
            lstm_layers=lstm_layers,
            batch_size=batch_size,
            add_encoders=create_encoders(),
            dropout=dropout,
            loss_fn=loss_fn,
            use_static_covariates=use_static_covariates,
            add_relative_index=add_relative_index,
            pl_trainer_kwargs=pl_trainer_kwargs
        )

        train_tss, val_tss = zip(*[sm.split_after(0.7) for sm in sms])
        # train the model
        model.fit(
            series=train_tss,
            past_covariates=wats,  # actuals
            future_covariates=wfts,  # forecasts
            val_series=sms,
            val_past_covariates=wats,  # actuals
            val_future_covariates=wfts,  # forecasts
            trainer=pl.Trainer(),
            # max_samples_per_ts=MAX_SAMPLES_PER_TS,
            num_loader_workers=num_workers,
        )

        # Evaluate how good it is on the validation set
        preds = model.predict(series=train_tss,
                              past_covariates=wats,
                              future_covariates=wfts, n=96)

        smapes = smape(val_tss, preds, n_jobs=-1, verbose=True)
        smape_val = np.mean(smapes)

        return smape_val if smape_val != np.nan else float("inf")

    return objective


def main_finetune(smart_meter_files: list[str], weather_forecast_files: list[str], weather_actuals_files: list[str],
                  model_config_path: str, save_model_path: str):
    logging.info('Starting training!')
    with open(model_config_path, 'r') as file:
        logging.info(f'Loading config from {model_config_path}')
        model_config = yaml.safe_load(file)

    # Creating folder to safe scalers and model 'YYYYMMDD_HHMM'
    folder_name = dt.datetime.now().strftime('%Y%m%d_%H%M')
    path = os.path.join('.', f'{folder_name}_{save_model_path}')
    os.makedirs(path)
    logging.info(f"Saving everything related to this model training run at: {path}")

    sm, wf, wa = load_csvs(model_config, smart_meter_files, weather_forecast_files, weather_actuals_files)
    smart_meter_tss, weather_forecast_ts, weather_actuals_ts = create_timeseries_from_dataframes(sm, wf, wa,
                                                                                                 scale=True,
                                                                                                 pickled_scaler_folder=path)

    objective = create_objective(model_config, data=(smart_meter_tss, weather_forecast_ts, weather_actuals_ts))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, timeout=7200, callbacks=[print_callback])
    # to limit the number of trials instead:
    # study.optimize(objective, n_trials=100, callbacks=[print_callback])

    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

    # only run this locally
    # plot_optimization_history(study)
    # plot_contour(study, params=["lr", "num_filters"])
    # plot_param_importances(study)


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

    smd_files, wfd_files, wad_files = [], [], []
    if args.smart_meter_data:
        smd_files = [file.strip() for file in ','.join(args.smart_meter_data).split(',')]
    if args.weather_forecast_data:
        wfd_files = [file.strip() for file in ','.join(args.weather_forecast_data).split(',')]
    if args.weather_actuals_data:
        wad_files = [file.strip() for file in ','.join(args.weather_actuals_data).split(',')]

    main_finetune(smd_files, wfd_files, wad_files, args.model_configuration, args.save_model_as)
