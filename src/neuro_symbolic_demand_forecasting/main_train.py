import argparse
import logging
import os
import pickle
from typing import Tuple, List, Optional, Dict
import torch.nn as nn
import numpy as np
import datetime as dt
import pandas as pd
import torch
import yaml
from darts import TimeSeries
from darts.models import RNNModel, TFTModel
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping

from neuro_symbolic_demand_forecasting.darts.custom_modules import ExtendedTFTModel, ExtendedRNNModel
from neuro_symbolic_demand_forecasting.darts.loss import CustomLoss

from sklearn.preprocessing import MinMaxScaler

from neuro_symbolic_demand_forecasting.encoders.encoders import AMS_TZ, create_encoders, LSTM_MAPPING, TFT_MAPPING


def load_csvs(_config: dict, _smart_meter_files: list[str], _weather_forecast_files: list[str],
              _weather_actuals_files: list[str]) -> Tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    sms = [pd.read_csv(s, index_col=None, parse_dates=['readingdate']) \
               .set_index('readingdate') for s in
           _smart_meter_files]
    for i, s in enumerate(sms):
        sms[i].index = s.index.tz_convert(AMS_TZ)

    wf = pd.read_csv(_weather_forecast_files[0], index_col=None, parse_dates=['valid_datetime']).set_index(
        'valid_datetime')
    wf.index = wf.index.tz_localize(None).tz_localize(AMS_TZ)
    wf = wf[_config['weather_forecast_features']]
    wf = wf.resample('15min').ffill()

    wa = pd.read_csv(_weather_actuals_files[0], index_col=None, parse_dates=['datetime_from']).set_index(
        'datetime_from')
    wa.index = wa.index.tz_localize(None).tz_localize(AMS_TZ)
    wa = wa[_config['weather_actuals_features']]
    wa = wa.resample('15min').mean()

    return sms, wf, wa


def _adjust_start_date(sm_ts: TimeSeries, min_weather, min_actuals) -> TimeSeries:
    min_smart_meter = sm_ts.time_index.min()
    overall_min_date = max(min_smart_meter, min_weather, min_actuals)
    logging.info(f'Adjusting start dates of smart meter timeseries from {min_smart_meter} to {overall_min_date}')
    # set index to the latest start (min) date
    return sm_ts[sm_ts.get_index_at_point(overall_min_date):]


def get_trainer_kwargs(_model_config: dict, callbacks: list) -> Tuple[dict, int]:
    early_stopper = EarlyStopping("val_loss", min_delta=0.0001, patience=10, verbose=True)
    if len(callbacks) == 0:
        callbacks = [early_stopper]
    else:
        callbacks.append(early_stopper)

    # detect if a GPU is available
    if torch.cuda.is_available() and _model_config['gpu_enable']:
        torch.set_float32_matmul_precision('high')
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "devices": [0],
            "callbacks": callbacks,
        }
        num_workers = 4
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}
        num_workers = 0
    return pl_trainer_kwargs, num_workers


def _init_model(_model_config: dict, _weights: Dict[str, float], callbacks: List[Callback], optimizer_kwargs=None):
    # throughout training we'll monitor the validation loss for early stopping
    pl_trainer_kwargs, num_workers = get_trainer_kwargs(_model_config, callbacks)
    encoders = create_encoders(_model_config['model_class'])

    match _model_config['model_class']:
        case "TFT":
            tft_config: dict = _model_config['tft_config']
            if not _model_config['run_learning_rate_finder'] and tft_config.get('optimizer_kwargs') is not None:
                optimizer_kwargs = tft_config['optimizer_kwargs']
            logging.info(f"Initiating the Temporal Fusion Transformer with these arguments: \n {tft_config}")
            if tft_config['loss_fn'] and tft_config['loss_fn'] == 'Custom':
                logging.info("Using TFTModel with Custom Module for custom Loss")
                logging.info(f"Initializing loss with these weights: {_weights}")
                return ExtendedTFTModel(
                    input_chunk_length=tft_config['input_chunk_length'],
                    output_chunk_length=tft_config['output_chunk_length'],
                    loss_fn=CustomLoss(TFT_MAPPING, _weights, {}),  # custom loss here
                    optimizer_kwargs=optimizer_kwargs,
                    add_encoders=encoders,
                    pl_trainer_kwargs=pl_trainer_kwargs,
                    **{k: v for k, v in tft_config.items() if
                       k not in ['input_chunk_length', 'output_chunk_length', 'loss_fn', 'optimizer_kwargs']}
                )
            else:
                logging.info("Using TFTModel with normal loss")
                return TFTModel(
                    input_chunk_length=tft_config['input_chunk_length'],
                    output_chunk_length=tft_config['output_chunk_length'],
                    optimizer_kwargs=optimizer_kwargs,
                    add_encoders=encoders,
                    loss_fn=nn.MSELoss(),
                    pl_trainer_kwargs=pl_trainer_kwargs,
                    **{k: v for k, v in tft_config.items() if
                       k not in ['input_chunk_length', 'output_chunk_length', 'optimizer_kwargs', 'loss_fn']}
                )
        case "LSTM":
            lstm_config: dict = _model_config['lstm_config']
            if not _model_config['run_learning_rate_finder'] and lstm_config.get('optimizer_kwargs') is not None:
                optimizer_kwargs = lstm_config['optimizer_kwargs']
            logging.info(f"Initiating the LSTM with these arguments: \n {lstm_config}")
            if lstm_config.get('loss_fn') == 'Custom':
                logging.info("Using LSTM with Custom Module for custom Loss")
                return ExtendedRNNModel(
                    model="LSTM",
                    loss_fn=CustomLoss(LSTM_MAPPING, _weights, thresholds={}),  # custom loss here
                    optimizer_kwargs=optimizer_kwargs,
                    add_encoders=encoders,
                    pl_trainer_kwargs=pl_trainer_kwargs,
                    **{k: v for k, v in lstm_config.items() if
                       k not in ['loss_fn', 'optimizer_kwargs']}
                )
            else:
                return RNNModel(
                    model="LSTM",
                    optimizer_kwargs=optimizer_kwargs,
                    add_encoders=encoders,
                    loss_fn=nn.MSELoss(),
                    pl_trainer_kwargs=pl_trainer_kwargs,
                    **{k: v for k, v in lstm_config.items() if k not in ['optimizer_kwargs', 'loss_fn']}
                )


def create_timeseries_from_dataframes(sm: List[pd.DataFrame], wf: pd.DataFrame, wa: pd.DataFrame,
                                      add_static_covariates: bool, scale: bool,
                                      pickled_scaler_folder: Optional[str] = None) -> Tuple[
    List[TimeSeries], List[TimeSeries], List[TimeSeries]]:
    # transforming to TimeSeries, using float32 is a bit more efficient than float64
    weather_forecast_ts = TimeSeries.from_dataframe(wf).astype(np.float32)
    weather_actuals_ts = TimeSeries.from_dataframe(wa).astype(np.float32)
    smart_meter_tss = [TimeSeries.from_dataframe(s).astype(np.float32) for s in sm]

    # scaling
    weather_forecast_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
    weather_actuals_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
    pv_smart_meter_scaler = Scaler(MinMaxScaler(feature_range=(-1, 1)))
    non_pv_smart_meter_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))

    non_pv_sms = [s for s in smart_meter_tss if any(s.min(axis=0) >= 0)]
    pv_sms = [s for s in smart_meter_tss if any(s.min(axis=0) < 0)]

    if scale:
        logging.info("Scaling ")
        weather_forecast_scaler = weather_forecast_scaler.fit(weather_forecast_ts)
        weather_forecast_ts = weather_forecast_scaler.transform(weather_forecast_ts)

        weather_actuals_scaler = weather_actuals_scaler.fit(weather_actuals_ts)
        weather_actuals_ts = weather_actuals_scaler.transform(weather_actuals_ts)

        pv_sms = pv_smart_meter_scaler.fit_transform(pv_sms)
        non_pv_sms = non_pv_smart_meter_scaler.fit_transform(non_pv_sms)
        smart_meter_tss = pv_sms + non_pv_sms

        if pickled_scaler_folder is not None:
            logging.info("Pickling the scalers for later use here.")
            with open(f'{pickled_scaler_folder}/scalers.pkl', 'wb') as f:
                # Pickle each object and write to the file
                pickle.dump(non_pv_smart_meter_scaler, f)
                pickle.dump(pv_smart_meter_scaler, f)
                pickle.dump(weather_actuals_scaler, f)
                pickle.dump(weather_forecast_scaler, f)

    if add_static_covariates:
        pv_static_covariate = pd.DataFrame(data={"is_pv": [1]})
        no_pv_static_covariate = pd.DataFrame(data={"is_pv": [0]})

        # .add_holidays(country_code='NL')
        pv_sms = [p.with_static_covariates(pv_static_covariate) for p in pv_sms]
        non_pv_sms = [p.with_static_covariates(no_pv_static_covariate) for p in
                      non_pv_sms]
        smart_meter_tss = pv_sms + non_pv_sms

    # matching the length of covariates with length of timeseries array
    weather_forecast_tss = [weather_forecast_ts for _ in sm]
    weather_actuals_tss = [weather_actuals_ts for _ in sm]

    smart_meter_tss = [_adjust_start_date(s, weather_forecast_ts.time_index.min(),
                                          weather_actuals_ts.time_index.min()) for s in smart_meter_tss]

    return smart_meter_tss, weather_forecast_tss, weather_actuals_tss


def main_train(smart_meter_files: list[str], weather_forecast_files: list[str], weather_actuals_files: list[str],
               model_config: dict, _path: str, _weights: Dict[str, float]):
    # load_dotenv()
    logging.info('Starting training!')
    # loading and bringing dataframes into appropriate shape and format
    sm, wf, wa = load_csvs(model_config, smart_meter_files, weather_forecast_files, weather_actuals_files)
    smart_meter_tss, weather_forecast_ts, weather_actuals_ts = create_timeseries_from_dataframes(sm, wf, wa,
                                                                                                 scale=True,
                                                                                                 add_static_covariates=True,
                                                                                                 pickled_scaler_folder=_path)
    # training
    model = _init_model(model_config, _weights, [], {})

    logging.info("Initialized model, beginning with fitting...")

    if model_config['run_with_validation']:
        input_chunk_length = model.input_chunk_length
        output_chunk_length = model.output_chunk_length
        validation_set_size = input_chunk_length + output_chunk_length
        train_tss, val_tss = zip(*[sm.split_after(len(sm) - validation_set_size - 1) for sm in smart_meter_tss])
        train_tss = list(train_tss)
        val_tss = list(val_tss)

        logging.info(f"Training with {[(t.time_index.min(), t.time_index.max()) for t in train_tss]}")
        logging.info(f"Validating with {[(t.time_index.min(), t.time_index.max()) for t in val_tss]}")

        # TFT
        if model.supports_past_covariates and model.supports_future_covariates:
            logging.info(f"Using static_covariates: {model.uses_static_covariates}")

            if model_config['run_learning_rate_finder']:
                results = model.lr_find(series=train_tss,
                                        past_covariates=weather_actuals_ts,
                                        val_series=val_tss,
                                        val_past_covariates=weather_actuals_ts,
                                        future_covariates=weather_forecast_ts,
                                        val_future_covariates=weather_forecast_ts
                                        )
                logging.info(f"Suggested Learning rate: {results.suggestion()}")
                model = _init_model(model_config, weights=_weights, callbacks=[],
                                    optimizer_kwargs={
                                        'lr': results.suggestion()})  # re-initialzing model with updated learning params

            model.fit(
                series=train_tss,
                past_covariates=weather_actuals_ts,
                future_covariates=weather_forecast_ts,
                val_series=val_tss,
                val_past_covariates=weather_actuals_ts,
                val_future_covariates=weather_forecast_ts,
                verbose=True,
            )
        # LSTM
        elif model.supports_future_covariates:
            logging.info("Initializing model with future_covariates")
            if model_config['run_learning_rate_finder']:
                results = model.lr_find(series=train_tss,
                                        val_series=val_tss,
                                        future_covariates=weather_forecast_ts,
                                        val_future_covariates=weather_forecast_ts
                                        )
                logging.info(f"Suggested Learning rate: {results.suggestion()}")
                # re-initialzing model with updated learning params
                model = _init_model(model_config, weights=_weights, callbacks=[],
                                    optimizer_kwargs={
                                        'lr': results.suggestion()})

            model.fit(
                train_tss,
                future_covariates=weather_forecast_ts,
                val_series=val_tss,
                val_future_covariates=weather_forecast_ts,
                verbose=True,
            )
        else:
            raise Exception(f'Training for other models not implemented yet')
    else:  # without validation
        # TFT without validation
        if model.supports_past_covariates and model.supports_future_covariates:
            if model_config['run_learning_rate_finder']:
                results = model.lr_find(series=smart_meter_tss,
                                        past_covariates=weather_actuals_ts,
                                        future_covariates=weather_forecast_ts)
                logging.info(f"Suggested Learning rate: {results.suggestion()}")
                model = _init_model(model_config, weights=_weights, callbacks=[],
                                    optimizer_kwargs={
                                        'lr': results.suggestion()})  # re-initialzing model with updated learning params
            model.fit(
                smart_meter_tss,
                past_covariates=weather_actuals_ts,
                future_covariates=weather_forecast_ts,
                verbose=True,
            )
        # LSTM without validation
        if model.supports_future_covariates:
            if model_config['run_learning_rate_finder']:
                results = model.lr_find(series=smart_meter_tss,
                                        future_covariates=weather_forecast_ts)
                logging.info(f"Suggested Learning rate: {results.suggestion()}")
                model = _init_model(model_config, weights=_weights, callbacks=[],
                                    optimizer_kwargs={
                                        'lr': results.suggestion()})  # re-initialzing model with updated learning params
            model.fit(
                smart_meter_tss,
                future_covariates=weather_forecast_ts,
                verbose=True,
            )
        else:
            raise Exception(f'Training for other models not implemented yet')

    logging.info(f"Saving model at {_path}/trained_model.pkl")
    model.save(f'{_path}/train_model.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model with smart meter data')
    parser.add_argument('-l', '--log_file', type=str, default=None, help='Path to the log file.')
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
    parser.add_argument('-w', '--weights', metavar='WEIGHTS', type=str,
                        help='String of space separated values for the weight initialization e.g. "1 0 0 0"')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.model_configuration, 'r') as file:
        logging.info(f'Loading config from {args.model_configuration}')
        model_config = yaml.safe_load(file)
        print(model_config)

    # Creating folder to save scalers and model 'YYYYMMDD_HHMM'
    current_datetime = dt.datetime.now().strftime('%Y%m%d_%H%M')
    base_dir, last_folder = os.path.split(args.save_model_as)
    new_last_folder = f"{current_datetime}_{last_folder}"
    path = os.path.join(base_dir, new_last_folder)
    os.makedirs(path)
    os.environ["MODEL_PATH"] = path

    logging.info(f"Saving everything related to this model training run at: {path}")

    smd_files, wfd_files, wad_files, weights = [], [], [], [0, 0, 0, 0]
    if args.smart_meter_data:
        smd_files = [file.strip() for file in ','.join(args.smart_meter_data).split(',')]
    if args.weather_forecast_data:
        wfd_files = [file.strip() for file in ','.join(args.weather_forecast_data).split(',')]
    if args.weather_actuals_data:
        wad_files = [file.strip() for file in ','.join(args.weather_actuals_data).split(',')]
    if args.weights:
        weights = [float(weight.strip()) for weight in args.weights.split(' ')]

    weight_names = ['no_neg_pred_night', 'no_neg_pred_nonpv', 'morning_evening_peaks', 'air_co']

    main_train(smd_files, wfd_files, wad_files, model_config, path, dict(zip(weight_names, weights)))
