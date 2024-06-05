import logging

import argparse
import os
import pickle
from typing import Sequence

import torch
import torch.nn as nn
import numpy as np
import optuna.pruners
import yaml
import datetime as dt
import pytorch_lightning as pl
from darts.metrics import smape, r2_score, rmse
from darts.models import TFTModel
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping

from neuro_symbolic_demand_forecasting.darts.custom_modules import ExtendedTFTModel
from neuro_symbolic_demand_forecasting.darts.loss import CustomLoss
from neuro_symbolic_demand_forecasting.encoders.encoders import create_encoders, TFT_MAPPING
from neuro_symbolic_demand_forecasting.main_train import load_csvs, create_timeseries_from_dataframes


# workaround to fix the "Expected parent" error: https://github.com/optuna/optuna/issues/4689
class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def print_callback(study, trials):
    logging.info(f"Best trials so far: {[(s._trial_id, s.params) for s in study.best_trials]} \n")
    # logging.info(f"Current value: {trial.value}, Current params: {trial.params}")
    # logging.info(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


def main_optimize(smart_meter_files: list[str], weather_forecast_files: list[str], weather_actuals_files: list[str],
                  _full_config: dict, _model_folder: str):
    sm, wf, wa = load_csvs(model_config, smart_meter_files, weather_forecast_files, weather_actuals_files)
    smart_meter_tss, weather_forecast_ts, weather_actuals_ts = create_timeseries_from_dataframes(sm, wf, wa,
                                                                                                 scale=True,
                                                                                                 add_static_covariates=True,
                                                                                                 pickled_scaler_folder=_model_folder)

    _config = model_config['tft_config']

    def objective(trial) -> Sequence[float]:

        # pruning_callback = OptunaPruning(trial, monitor="val_loss")

        trainer = pl.Trainer(
            max_epochs=_config['n_epochs'],
            callbacks=[EarlyStopping("val_loss", min_delta=0.001, patience=5, verbose=True)]
        )
        num_workers = 0
        if torch.cuda.is_available() and _full_config['gpu_enable']:
            torch.set_float32_matmul_precision('high')
            trainer = pl.Trainer(
                devices=[0],
                accelerator='gpu',
                max_epochs=_config['n_epochs'],
                callbacks=[EarlyStopping("val_loss", min_delta=0.001, patience=5, verbose=True)]
            )
            num_workers = _full_config['num_workers']

        def get_suggestion(suggest_fn, name):
            if _config[name]['increment'] == 0:
                return _config[name]['start']
            else:
                logging.info(f"Parameter {name} will be tuned.")
                return suggest_fn(name,
                                  low=_config[name]['start'],
                                  high=_config[name]['end'],
                                  step=_config[name]['increment'])

        input_chunk_length = get_suggestion(trial.suggest_int, 'input_chunk_length')
        hidden_size = get_suggestion(trial.suggest_int, 'hidden_size')
        hidden_continuous_size = get_suggestion(trial.suggest_int, 'hidden_continuous_size')
        num_attention_heads = get_suggestion(trial.suggest_int, 'num_attention_heads')
        lstm_layers = get_suggestion(trial.suggest_int, 'lstm_layers')
        batch_size = get_suggestion(trial.suggest_int, 'batch_size')
        dropout = get_suggestion(trial.suggest_float, 'dropout')

        if _config['loss_fn'] == 'Custom':
            weight_names = ['no_neg_pred_night', 'no_neg_pred_nonpv', 'morning_evening_peaks', 'air_co']
            non_neg_pred_night = get_suggestion(trial.suggest_float, 'weights_no_neg_pred_night')
            no_neg_pred_nonpv = get_suggestion(trial.suggest_float, 'weights_no_neg_pred_night')
            morning_evening_peaks = get_suggestion(trial.suggest_float, 'weights_morning_evening_peaks')
            air_co = get_suggestion(trial.suggest_float, 'weights_air_co')
            weights = dict(zip(weight_names, [non_neg_pred_night, no_neg_pred_nonpv, morning_evening_peaks, air_co]))
        else:
            weights = {}

        use_static_covariates = trial.suggest_categorical("use_static_covariates", [True])
        add_relative_index = trial.suggest_categorical("add_relative_index", [False])

        match _config.get('loss_fn'):
            case 'Custom':
                loss_fn = CustomLoss(TFT_MAPPING, weights, {})
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
        r2s = r2_score(val_tss, preds, n_jobs=-1, verbose=True)
        r2_val = np.mean(r2s)
        rmses = rmse(val_tss, preds, n_jobs=-1, verbose=True)
        rmse_val = np.mean(rmses)

        return rmse_val if rmse_val != np.nan else float("inf"), \
               smape_val if smape_val != np.nan else float("inf"), \
               r2_val if r2_val != np.nan else float("inf")

    study = optuna.create_study(study_name='test',
                                # minimize rmse, and smape, maximize r2
                                directions=["minimize", "minimize", "maximize"],
                                # sampler=RandomSampler()
                                )

    study.optimize(objective, n_jobs=_full_config['num_workers'], n_trials=_full_config['trials'], callbacks=[print_callback])

    logging.info(f"Best params: {[s.params for s in study.trials]}")
    logging.info(f"Best value: {[s.values for s in study.trials]}")
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

     # testing multi-obejctive
    # def objective(trial):
    #     x = trial.suggest_float("x", -100, 100)
    #     y = trial.suggest_categorical("y", [-1, 0, 1])
    #     f1 = x ** 2 + y
    #     f2 = -((x - 2) ** 2 + y)
    #     return f1, f2
    #
    #
    # # We minimize the first objective and maximize the second objective.
    # sampler = optuna.samplers.RandomSampler()
    # sstudy = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)
    # sstudy.optimize(objective, n_trials=100, callbacks=[print_callback])


