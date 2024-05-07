import argparse
import datetime as dt
import logging

import pandas as pd
import yaml
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, smape, mae

from neuro_symbolic_demand_forecasting.darts.loss import CustomLoss

from sklearn.preprocessing import MinMaxScaler


def _load_csvs(_config: dict, _smart_meter_files: list[str], _weather_forecast_files: list[str],
               _weather_actuals_files: list[str]):
    sm = pd.read_csv(_smart_meter_files[0], index_col=None, parse_dates=['readingdate']).set_index('readingdate')

    wf = pd.read_csv(_weather_forecast_files[0], index_col=None, parse_dates=['valid_datetime']).set_index(
        'valid_datetime')
    wf = wf[_config['weather_forecast_features']]
    wf = wf.resample('15min').ffill()

    wa = pd.read_csv(_weather_actuals_files[0], index_col=None, parse_dates=['datetime_from']).set_index(
        'datetime_from')
    wa = wa[_config['weather_actuals_features']]
    wa = wa.resample('15min').mean()

    return sm, wf, wa


def _adjust_start_date(sm_ts: TimeSeries, min_weather, min_actuals) -> TimeSeries:
    min_smart_meter = sm_ts.time_index.min()
    overall_min_date = max(min_smart_meter, min_weather, min_actuals)
    # set index to the latest start (min) date
    return sm_ts[sm_ts.get_index_at_point(overall_min_date):]


def _build_torch_kwargs(kwargs: dict) -> dict:
    if kwargs['loss_fn'] and kwargs['loss_fn'] == 'CustomLoss':
        logging.info('Loading in custom loss function')
        kwargs['loss_fn'] = CustomLoss()
    return kwargs


def _init_model(model_config: dict):
    match model_config['model_class']:
        case "TFT":
            tft_config: dict = model_config['tft_config']
            logging.info(f"Initiating the Temporal Fusion Transformer with these arguments: \n {tft_config}")
            return TFTModel(
                input_chunk_length=tft_config['input_chunk_length'],
                output_chunk_length=tft_config['output_chunk_length'],
                **{k: v for k, v in tft_config.items() if
                   k not in ['input_chunk_length', 'output_chunk_length', 'torch_kwargs']},
                **_build_torch_kwargs(tft_config['torch_kwargs'])
            )


def main_train(smart_meter_files: list[str], weather_forecast_files: list[str], weather_actuals_files: list[str],
               model_config_path: str, save_validation_forecast: bool):
    # load_dotenv()
    logging.info('Starting training!')
    with open(model_config_path, 'r') as file:
        logging.info(f'Loading config from {model_config_path}')
        model_config = yaml.safe_load(file)

    # loading
    sm, wf, wa = _load_csvs(model_config, smart_meter_files, weather_forecast_files, weather_actuals_files)

    # scaling
    weather_forecast_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
    weather_actuals_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
    smart_meter_scaler = Scaler(MinMaxScaler(feature_range=(-1, 1)))

    weather_forecast_ts = weather_forecast_scaler.fit_transform(TimeSeries.from_dataframe(wf))
    weather_actuals_ts = weather_actuals_scaler.fit_transform(TimeSeries.from_dataframe(wa))
    smart_meter_ts = smart_meter_scaler.fit_transform(TimeSeries.from_dataframe(sm))
    smart_meter_ts = _adjust_start_date(smart_meter_ts, weather_forecast_ts.time_index.min(),
                                        weather_actuals_ts.time_index.min())

    # training
    model = _init_model(model_config)
    train_meter, val_meter = smart_meter_ts.split_after(0.8)
    model.fit(
        train_meter,
        past_covariates=weather_actuals_ts,
        future_covariates=weather_forecast_ts,
        val_series=val_meter,
        val_past_covariates=weather_actuals_ts,
        val_future_covariates=weather_forecast_ts,
        verbose=True,
        # trainer=pl_trainer_kwargs # would be nice to have early stopping here
    )
    # validate

    forecast, actual = val_meter[:-96], val_meter[-96:]
    # # train_meter.plot()
    # val_meter.plot()
    #
    pred = model.predict(n=96, series=forecast, past_covariates=weather_actuals_ts,
                         future_covariates=weather_forecast_ts)

    logging.info("MAPE = {:.2f}%".format(mape(actual, pred)))
    logging.info("SMAPE = {:.2f}%".format(smape(actual, pred)))
    logging.info("MAE = {:.2f}%".format(mae(actual, pred)))


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
    args = parser.parse_args()

    if args.log_file:
        logging.basicConfig(level=logging.INFO, filename=args.log_file)
    else:
        logging.basicConfig(level=logging.INFO)

    smd_files, wfd_files, wad_files = [], [], []
    if args.smart_meter_data:
        smd_files = [file.strip() for file in ','.join(args.smart_meter_data).split(',')]
    if args.weather_forecast_data:
        wfd_files = [file.strip() for file in ','.join(args.weather_forecast_data).split(',')]
    if args.weather_actuals_data:
        wad_files = [file.strip() for file in ','.join(args.weather_actuals_data).split(',')]

    save_validation_forecast = True

    main_train(smd_files, wfd_files, wad_files, args.model_configuration, save_validation_forecast=True)
