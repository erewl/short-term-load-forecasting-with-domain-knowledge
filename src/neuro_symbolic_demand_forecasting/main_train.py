import argparse
import logging
from typing import Tuple

import pandas as pd
import yaml
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, smape, mae
from neuro_symbolic_demand_forecasting.darts.custom_modules import ExtendedTFTModel
from neuro_symbolic_demand_forecasting.darts.loss import CustomLoss

from sklearn.preprocessing import MinMaxScaler


def _load_csvs(_config: dict, _smart_meter_files: list[str], _weather_forecast_files: list[str],
               _weather_actuals_files: list[str]) -> Tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    sms = [pd.read_csv(s, index_col=None, parse_dates=['readingdate']).set_index('readingdate') for s in
           _smart_meter_files]

    wf = pd.read_csv(_weather_forecast_files[0], index_col=None, parse_dates=['valid_datetime']).set_index(
        'valid_datetime')
    wf = wf[_config['weather_forecast_features']]
    wf = wf.resample('15min').ffill()

    wa = pd.read_csv(_weather_actuals_files[0], index_col=None, parse_dates=['datetime_from']).set_index(
        'datetime_from')
    wa = wa[_config['weather_actuals_features']]
    wa = wa.resample('15min').mean()

    return sms, wf, wa


def _adjust_start_date(sm_ts: TimeSeries, min_weather, min_actuals) -> TimeSeries:
    min_smart_meter = sm_ts.time_index.min()
    overall_min_date = max(min_smart_meter, min_weather, min_actuals)
    logging.info(f'Adjusting start dates of smart meter timeseries from {min_smart_meter} to {overall_min_date}')
    # set index to the latest start (min) date
    return sm_ts[sm_ts.get_index_at_point(overall_min_date):]


def _build_torch_kwargs(kwargs: dict) -> dict:
    if kwargs['loss_fn'] and kwargs['loss_fn'] == 'CustomLoss':
        logging.info('Loading in custom loss function')
        # init of custom loss
        kwargs['loss_fn'] = CustomLoss([0.1])
    return kwargs


def _init_model(model_config: dict):
    match model_config['model_class']:
        case "TFT":
            tft_config: dict = model_config['tft_config']
            logging.info(f"Initiating the Temporal Fusion Transformer with these arguments: \n {tft_config}")
            return ExtendedTFTModel(
                input_chunk_length=tft_config['input_chunk_length'],
                output_chunk_length=tft_config['output_chunk_length'],
                # loss_fn=CustomLoss([0.1]),  # custom loss here
                # pl_trainer_kwargs={
                #     "accelerator": "gpu",
                #     "devices": [0]
                # },
                **{k: v for k, v in tft_config.items() if
                   k not in ['input_chunk_length', 'output_chunk_length', 'loss_fn']}
            )
        case "LSTM":
            lstm_config: dict = model_config['lstm_config']
            logging.info(f"Initiating the Temporal Fusion Transformer with these arguments: \n {lstm_config}")
            return RNNModel(
                model="LSTM",
                loss_fn=CustomLoss([0.1]),  # custom loss here
                **{k: v for k, v in lstm_config.items() if
                   k not in ['loss_fn']}
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
    smart_meter_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))

    weather_forecast_ts = weather_forecast_scaler.fit_transform(TimeSeries.from_dataframe(wf))
    weather_actuals_ts = weather_actuals_scaler.fit_transform(TimeSeries.from_dataframe(wa))
    smart_meter_tss = smart_meter_scaler.fit_transform([TimeSeries.from_dataframe(s) for s in sm])

    logging.info("Dtypes of SM, WF, WA")
    logging.info(smart_meter_tss[0].values().dtype)
    logging.info(weather_forecast_ts.values().dtype)
    logging.info(weather_actuals_ts.values().dtype)


    # creating a series of the same length as smart_meter_ts
    weather_forecast_tss = [weather_forecast_ts for _ in sm]
    weather_actuals_tss = [weather_actuals_ts for _ in sm]
    smart_meter_tss = [_adjust_start_date(s, weather_forecast_ts.time_index.min(),
                                          weather_actuals_ts.time_index.min()) for s in smart_meter_tss]
    # smart_meter_tss = [s.add_datetime_attribute('weekday', one_hot=True) for s in smart_meter_tss]

    # training
    model = _init_model(model_config)
    logging.info("Initialized model, beginning with fitting...")
    logging.info(vars(model))
    train_tss, val_tss = zip(*[sm.split_after(0.7) for sm in smart_meter_tss])
    match model_config['model_class']:
        case 'LSTM':
            model.fit(
                train_tss,
                future_covariates=weather_forecast_tss,
                val_series=val_tss,
                # val_past_covariates=weather_actuals_ts,
                # val_future_covariates=weather_forecast_ts,
                verbose=True,
                # trainer=pl_trainer_kwargs # would be nice to have early stopping here
            )
        case 'TFT':
            model.fit(
                train_tss,
                past_covariates=weather_actuals_tss,
                future_covariates=weather_forecast_tss,
                val_series=val_tss,
                val_past_covariates=weather_actuals_ts,
                val_future_covariates=weather_forecast_ts,
                verbose=True,
                # trainer=pl_trainer_kwargs # would be nice to have early stopping here
            )
        case other:
            raise Exception(f'Training for {other} not implemented yet')


    # validate
    forecast, actual = smart_meter_tss[0][:-96], smart_meter_tss[0][-96:]
    # # train_meter.plot()
    # val_meter.plot()
    #
    pred = model.predict(n=96, series=forecast,
                         future_covariates=weather_forecast_ts)

    logging.info("MAPE = {:.2f}%".format(mape(actual, pred)))
    logging.info("SMAPE = {:.2f}%".format(smape(actual, pred)))
    logging.info("MAE = {:.2f}%".format(mae(actual, pred)))

    model.save('./2024-05-13_tft_model_baseline.pkl')


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
        print("No log file specified, writing to stdout")
        logging.basicConfig(level=logging.DEBUG)

    smd_files, wfd_files, wad_files = [], [], []
    if args.smart_meter_data:
        smd_files = [file.strip() for file in ','.join(args.smart_meter_data).split(',')]
    if args.weather_forecast_data:
        wfd_files = [file.strip() for file in ','.join(args.weather_forecast_data).split(',')]
    if args.weather_actuals_data:
        wad_files = [file.strip() for file in ','.join(args.weather_actuals_data).split(',')]

    save_validation_forecast = True

    main_train(smd_files, wfd_files, wad_files, args.model_configuration, save_validation_forecast=True)
