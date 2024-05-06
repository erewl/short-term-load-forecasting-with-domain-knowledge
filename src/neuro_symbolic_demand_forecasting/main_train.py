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


def main_train(smart_meter_files: list[str], weather_forecast_files: list[str], weather_actuals_files: list[str],
               model_config_path: str, save_validation_forecast: bool):
    # load_dotenv()
    with open(model_config_path, 'r') as file:
        logging.info(f'Loading config from {model_config_path}')
        model_config = yaml.safe_load(file)

    sm = pd.read_csv(smart_meter_files[0], index_col=None, parse_dates=['readingdate']).set_index('readingdate')

    wf = pd.read_csv(weather_forecast_files[0], index_col=None, parse_dates=['valid_datetime']).set_index(
        'valid_datetime')
    wf = wf[model_config['weather_forecast_features']]
    wf = wf.resample('15min').ffill()

    wa = pd.read_csv(weather_actuals_files[0], index_col=None, parse_dates=['datetime_from']).set_index('datetime_from')
    wa = wa[model_config['weather_actuals_features']]
    wa = wa.resample('15min').mean()

    # if we are training on the solar cluster we want -1 to 1 range, else just 0 to 1
    smart_meter_scaler = Scaler(MinMaxScaler(feature_range=(-1, 1)))
    smart_meter_ts = smart_meter_scaler.fit_transform(TimeSeries.from_dataframe(sm))

    weather_forecast_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
    weather_forecast_ts = weather_forecast_scaler.fit_transform(TimeSeries.from_dataframe(wf))

    weather_actuals_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
    weather_actuals_ts = weather_actuals_scaler.fit_transform(TimeSeries.from_dataframe(wa))

    # validating datasets and (eventually adjusting)
    logging.info(f"{len(smart_meter_ts)}, {len(weather_forecast_ts)}, {len(weather_actuals_ts)}")
    # checking that train_meter is surrounded by the covariates
    # print(train_meter.time_index.min(), train_meter.time_index.max())
    # print(weather_ts.time_index.min(), weather_ts.time_index.max())
    # print(actuals_ts.time_index.min(), actuals_ts.time_index.max())

    model = init_model(model_config)

    # train_meter, val_meter = meter_ts.split_after(0.8)
    model.fit(
        smart_meter_ts,
        past_covariates=weather_actuals_ts,
        future_covariates=weather_forecast_ts,
        # val_series=val_meter,
        # val_past_covariates=ats,
        # val_future_covariates=wts,
        verbose=True,
        # trainer=pl_trainer_kwargs
    )
    # validate

    # forecast, actual = val_meter[:-96], val_meter[-96:]
    # # train_meter.plot()
    # val_meter.plot()
    #
    # pred = model.predict(n=500, series=forecast, past_covariates=ats, future_covariates=wts)
    pred = []
    actual = []

    print("MAPE = {:.2f}%".format(mape(actual, pred)))
    print("SMAPE = {:.2f}%".format(smape(actual, pred)))
    print("MAE = {:.2f}%".format(mae(actual, pred)))


def build_torch_kwargs(kwargs: dict) -> dict:
    if kwargs['loss_fn'] and kwargs['loss_fn'] == 'CustomLoss':
        logging.info('Loading in custom loss function')
        kwargs['loss_fn'] = CustomLoss()
    return kwargs


def init_model(model_config: dict):
    match model_config['model_class']:
        case "TFT":
            tft_config: dict = model_config['tft_config']
            logging.info(f"Initiating the Temporal Fusion Transformer with these arguments: \n {tft_config}")
            return TFTModel(
                input_chunk_length=tft_config['input_chunk_length'],
                output_chunk_length=tft_config['output_chunk_length'],
                **{k: v for k, v in tft_config.items() if
                   k not in ['input_chunk_length', 'output_chunk_length', 'torch_kwargs']},
                **build_torch_kwargs(tft_config['torch_kwargs'])
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Train the model with smart meter data')
    parser.add_argument('-smd', '--smart-meter-data', metavar='SMD_FILES', type=str, nargs='+',
                        help='comma-separated list of smart meter data csv files to be used for training')
    parser.add_argument('-wfd', '--weather-forecast-data', metavar='WFD_FILES', type=str, nargs='+',
                        help='comma-separated list of weather forecast csv files to be used for training')
    parser.add_argument('-wad', '--weather-actuals-data', metavar='WAD_FILES', type=str, nargs='+',
                        help='comma-separated list of weather actuals csv files to be used for training')
    parser.add_argument('-md', '--model-configuration', metavar='MODEL_CONFIG_PATH', type=str,
                        help='path to the model configuration YAML file')
    args = parser.parse_args()

    smd_files, wfd_files, wad_files = [], [], []
    if args.smart_meter_data:
        smd_files = [file.strip() for file in ','.join(args.smart_meter_data).split(',')]
    if args.weather_forecast_data:
        wfd_files = [file.strip() for file in ','.join(args.weather_forecast_data).split(',')]
    if args.weather_actuals_data:
        wad_files = [file.strip() for file in ','.join(args.weather_actuals_data).split(',')]

    save_validation_forecast = True

    logging.info('Starting training!')
    main_train(smd_files, wfd_files, wad_files, args.model_configuration, save_validation_forecast=True)
