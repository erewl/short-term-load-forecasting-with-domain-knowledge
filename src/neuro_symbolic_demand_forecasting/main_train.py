import argparse
import logging
import os
import pickle
from typing import Tuple, List, Optional

import numpy as np
import pytz
from astral.sun import sun
from astral import LocationInfo
import datetime as dt
import pandas as pd
import torch
import yaml
from darts import TimeSeries
from darts.models import RNNModel, TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, smape, mae
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping

from neuro_symbolic_demand_forecasting.darts.custom_modules import ExtendedTFTModel, ExtendedRNNModel
from neuro_symbolic_demand_forecasting.darts.loss import CustomLoss

from sklearn.preprocessing import MinMaxScaler

# timezones
AMS_TZ = pytz.timezone('Europe/Amsterdam')


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


def encode_ptu(idx):
    total_minutes = idx.hour * 60 + idx.minute
    ptu = total_minutes // 15
    return ptu + 1


def create_encoders() -> dict:
    encoders = {
        'cyclic': {'future': ['month', 'day']},
        'datetime_attribute': {'future': ['dayofweek'], 'past': ['dayofweek']},
        'custom': {'past': [encode_ptu], 'future': [encode_ptu]},
        # 'position': {'past': ['relative'], 'future': ['relative']},
        'transformer': Scaler(),
        'tz': 'Europe/Amsterdam'
    }
    return encoders


def _adjust_start_date(sm_ts: TimeSeries, min_weather, min_actuals) -> TimeSeries:
    min_smart_meter = sm_ts.time_index.min()
    overall_min_date = max(min_smart_meter, min_weather, min_actuals)
    logging.info(f'Adjusting start dates of smart meter timeseries from {min_smart_meter} to {overall_min_date}')
    # set index to the latest start (min) date
    return sm_ts[sm_ts.get_index_at_point(overall_min_date):]


def get_trainer_kwargs(_model_config: dict, callbacks: list) -> Tuple[dict, int]:
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    if len(callbacks) == 0:
        callbacks = [early_stopper]
    else:
        callbacks.append(early_stopper)

    # detect if a GPU is available
    if torch.cuda.is_available() and _model_config['gpu_enable']:
        torch.set_float32_matmul_precision('high')
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "gpus": -1,
            "auto_select_gpus": True,
            "callbacks": callbacks,
        }
        num_workers = 4
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}
        num_workers = 0
    return pl_trainer_kwargs, num_workers


def _init_model(model_config: dict, callbacks: List[Callback], optimizer_kwargs=None) -> TorchForecastingModel:
    # throughout training we'll monitor the validation loss for early stopping
    pl_trainer_kwargs = get_trainer_kwargs(model_config, callbacks)
    encoders = create_encoders()

    match model_config['model_class']:
        case "TFT":
            tft_config: dict = model_config['tft_config']
            if not model_config['run_learning_rate_finder'] and tft_config.get('optimizer_kwargs') is not None:
                optimizer_kwargs = tft_config['optimizer_kwargs']
            logging.info(f"Initiating the Temporal Fusion Transformer with these arguments: \n {tft_config}")
            if tft_config['loss_fn'] and tft_config['loss_fn'] == 'Custom':
                logging.info("Using TFTModel with Custom Module for custom Loss")
                return ExtendedTFTModel(
                    input_chunk_length=tft_config['input_chunk_length'],
                    output_chunk_length=tft_config['output_chunk_length'],
                    loss_fn=CustomLoss(),  # custom loss here
                    optimizer_kwargs=optimizer_kwargs,
                    add_encoders=encoders,
                    pl_trainer_kwargs=pl_trainer_kwargs,
                    **{k: v for k, v in tft_config.items() if
                       k not in ['input_chunk_length', 'output_chunk_length', 'loss_fn', 'optimizer_kwargs']}
                )
            else:
                return TFTModel(
                    input_chunk_length=tft_config['input_chunk_length'],
                    output_chunk_length=tft_config['output_chunk_length'],
                    optimizer_kwargs=optimizer_kwargs,
                    add_encoders=encoders,
                    pl_trainer_kwargs=pl_trainer_kwargs,
                    **{k: v for k, v in tft_config.items() if
                       k not in ['input_chunk_length', 'output_chunk_length', 'optimizer_kwargs']}
                )
        case "LSTM":
            lstm_config: dict = model_config['lstm_config']
            if not model_config['run_learning_rate_finder'] and lstm_config.get('optimizer_kwargs') is not None:
                optimizer_kwargs = lstm_config['optimizer_kwargs']
            logging.info(f"Initiating the LSTM with these arguments: \n {lstm_config}")
            if lstm_config.get('loss_fn') == 'Custom':
                logging.info("Using TFTModel with Custom Module for custom Loss")
                return ExtendedRNNModel(
                    model="LSTM",
                    loss_fn=CustomLoss(),  # custom loss here
                    optimizer_kwargs=optimizer_kwargs,
                    pl_trainer_kwargs=pl_trainer_kwargs,
                    **{k: v for k, v in lstm_config.items() if
                       k not in ['loss_fn', 'optimizer_kwargs']}
                )
            else:
                return RNNModel(
                    model="LSTM",
                    optimizer_kwargs=optimizer_kwargs,
                    pl_trainer_kwargs=pl_trainer_kwargs,
                    **{k: v for k, v in lstm_config.items() if k not in ['optimizer_kwargs']}
                )


def get_part_of_day(s, _city: LocationInfo):
    i = s.name
    sun_info = sun(_city.observer, date=i.date())
    sunrise = sun_info['sunrise'].astimezone(AMS_TZ).time()
    sunset = sun_info['sunset'].astimezone(AMS_TZ).time()

    if i.time() < sunrise or i.time() > sunset:
        return 'Night'
    elif sunrise <= i.time() < dt.time(9, 0, 0):
        return 'Morning'
    elif dt.time(9, 0, 0) <= i.time() < dt.time(13, 0, 0):
        return 'Midday'
    elif dt.time(13, 0, 0) <= i.time() < dt.time(17, 0, 0):
        return 'Afternoon'
    else:
        return 'Evening'


def create_timeseries_from_dataframes(sm: List[pd.DataFrame], wf: pd.DataFrame, wa: pd.DataFrame, scale: bool,
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
    non_pv_smart_meter_scaler = Scaler(MinMaxScaler(feature_range=(-1, 1)))

    if scale:
        logging.info("Scaling ")
        weather_forecast_ts = weather_forecast_scaler.fit_transform(weather_forecast_ts)
        weather_actuals_ts = weather_actuals_scaler.fit_transform(weather_actuals_ts)
        smart_meter_tss = pv_smart_meter_scaler.fit_transform(smart_meter_tss)
        if pickled_scaler_folder is not None:
            logging.info("Pickling the scalers for later use here.")
            with open(f'{pickled_scaler_folder}/scalers.pkl', 'wb') as f:
                # Pickle each object and write to the file
                pickle.dump(non_pv_smart_meter_scaler, f)
                pickle.dump(pv_smart_meter_scaler, f)
                pickle.dump(weather_actuals_scaler, f)
                pickle.dump(weather_forecast_scaler, f)

    # matching the length of covariates with length of timeseries array
    weather_forecast_tss = [weather_forecast_ts for _ in sm]
    weather_actuals_tss = [weather_actuals_ts for _ in sm]

    smart_meter_tss = [_adjust_start_date(s, weather_forecast_ts.time_index.min(),
                                          weather_actuals_ts.time_index.min()) for s in smart_meter_tss]

    return smart_meter_tss, weather_forecast_tss, weather_actuals_tss


def main_train(smart_meter_files: list[str], weather_forecast_files: list[str], weather_actuals_files: list[str],
               model_config_path: str, save_model_path: str):
    # load_dotenv()
    logging.info('Starting training!')
    with open(model_config_path, 'r') as file:
        logging.info(f'Loading config from {model_config_path}')
        model_config = yaml.safe_load(file)

    city = LocationInfo("Amsterdam", "Netherlands", "Europe/Amsterdam")

    # Creating folder to safe scalers and model 'YYYYMMDD_HHMM'
    folder_name = dt.datetime.now().strftime('%Y%m%d_%H%M')
    path = os.path.join('.', f'{folder_name}_{save_model_path}')
    os.makedirs(path)
    logging.info(f"Saving everything related to this model training run at: {path}")

    # loading and bringing dataframes into appropriate shape and format
    sm, wf, wa = load_csvs(model_config, smart_meter_files, weather_forecast_files, weather_actuals_files)
    smart_meter_tss, weather_forecast_ts, weather_actuals_ts = create_timeseries_from_dataframes(sm, wf, wa,
                                                                                                 scale=True,
                                                                                                 pickled_scaler_folder=path)
    logging.info("Dtypes of SM, WF, WA")
    logging.info(smart_meter_tss[0].values().dtype)
    logging.info(weather_forecast_ts[0].values().dtype)
    logging.info(weather_actuals_ts[0].values().dtype)

    # training
    model = _init_model(model_config, [], {})
    logging.info("Initialized model, beginning with fitting...")

    if model_config['run_with_validation']:
        train_tss, val_tss = zip(*[sm.split_after(0.7) for sm in smart_meter_tss])

        # TFT
        if model.supports_past_covariates and model.supports_future_covariates:
            if model_config['run_learning_rate_finder']:
                results = model.lr_find(series=train_tss,
                                        past_covariates=weather_actuals_ts,
                                        val_series=val_tss,
                                        val_past_covariates=weather_actuals_ts,
                                        future_covariates=weather_forecast_ts,
                                        val_future_covariates=weather_forecast_ts
                                        )
                logging.info(f"Suggested Learning rate: {results.suggestion()}")
                model = _init_model(model_config, callbacks=[],
                                    optimizer_kwargs={
                                        'lr': results.suggestion()})  # re-initialzing model with updated learning params
            model.fit(
                train_tss,
                past_covariates=weather_actuals_ts,
                future_covariates=weather_forecast_ts,
                val_series=val_tss,
                val_past_covariates=weather_actuals_ts,
                val_future_covariates=weather_forecast_ts,
                verbose=True,
                # trainer=pl_trainer_kwargs # would be nice to have early stopping here
            )

        # LSTM
        if model.supports_future_covariates:
            if model_config['run_learning_rate_finder']:
                results = model.lr_find(series=train_tss,
                                        val_series=val_tss,
                                        future_covariates=weather_forecast_ts,
                                        val_future_covariates=weather_forecast_ts
                                        )
                logging.info(f"Suggested Learning rate: {results.suggestion()}")
                # re-initialzing model with updated learning params
                model = _init_model(model_config, callbacks=[],
                                    optimizer_kwargs={
                                        'lr': results.suggestion()})
            model.fit(
                train_tss,
                future_covariates=weather_forecast_ts,
                val_series=val_tss,
                # val_past_covariates=weather_actuals_ts, # doesnt support past covariates
                val_future_covariates=weather_forecast_ts,
                verbose=True,
                # trainer=pl_trainer_kwargs # would be nice to have early stopping here
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
                model = _init_model(model_config, callbacks=[],
                                    optimizer_kwargs={
                                        'lr': results.suggestion()})  # re-initialzing model with updated learning params
            model.fit(
                smart_meter_tss,
                past_covariates=weather_actuals_ts,
                future_covariates=weather_forecast_ts,
                verbose=True,
                # trainer=pl_trainer_kwargs # would be nice to have early stopping here
            )
        # LSTM without validation
        if model.supports_future_covariates:
            if model_config['run_learning_rate_finder']:
                results = model.lr_find(series=smart_meter_tss,
                                        future_covariates=weather_forecast_ts)
                logging.info(f"Suggested Learning rate: {results.suggestion()}")
                model = _init_model(model_config, callbacks=[],
                                    optimizer_kwargs={
                                        'lr': results.suggestion()})  # re-initialzing model with updated learning params
            model.fit(
                smart_meter_tss,
                future_covariates=weather_forecast_ts,
                verbose=True,
                # trainer=pl_trainer_kwargs # would be nice to have early stopping here
            )
        else:
            raise Exception(f'Training for other models not implemented yet')

    # validate
    forecast, actual = smart_meter_tss[0][:-96], smart_meter_tss[0][-96:]

    pred = model.predict(n=96, series=forecast,
                         future_covariates=weather_forecast_ts)

    logging.info("MAPE = {:.2f}%".format(mape(actual, pred)))
    logging.info("SMAPE = {:.2f}%".format(smape(actual, pred)))
    logging.info("MAE = {:.2f}%".format(mae(actual, pred)))

    logging.info(f"Saving model at {path}/trained_model.pkl")
    model.save(f'{path}/train_model.pkl')


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    smd_files, wfd_files, wad_files = [], [], []
    if args.smart_meter_data:
        smd_files = [file.strip() for file in ','.join(args.smart_meter_data).split(',')]
    if args.weather_forecast_data:
        wfd_files = [file.strip() for file in ','.join(args.weather_forecast_data).split(',')]
    if args.weather_actuals_data:
        wad_files = [file.strip() for file in ','.join(args.weather_actuals_data).split(',')]

    main_train(smd_files, wfd_files, wad_files, args.model_configuration, args.save_model_as)
