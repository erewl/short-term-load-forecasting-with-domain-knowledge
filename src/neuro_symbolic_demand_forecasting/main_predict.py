import logging
import datetime as dt

import argparse
import pickle

import yaml
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from neuro_symbolic_demand_forecasting.main_train import load_csvs, create_timeseries_from_dataframes, AMS_TZ


def main_evaluate(_config: dict, start_date_time: dt.datetime, end_date_time: dt.datetime,
                  smart_meter_data_files: list[str],
                  weather_forecast_file: str, weather_actuals_file: str, model_folder: str):
    sms, wf, wa = load_csvs(_config, smart_meter_data_files, [weather_forecast_file], [weather_actuals_file])

    sm_tss, wf_ts, wa_ts = create_timeseries_from_dataframes(sms, wf, wa, scale=False)

    # initializing unfitted scalers that are getting loaded from a pickle in this step
    with open(model_folder + '/scalers.pkl', 'rb') as f:
        non_pv_scaler = pickle.load(f)
        pv_scaler = pickle.load(f)
        wa_scaler = pickle.load(f)
        wf_scaler = pickle.load(f)

    sm_tss = [pv_scaler.transform(s) for s in sm_tss]
    wf_ts = wf_scaler.transform(wf_ts)
    wa_ts = wa_scaler.transform(wa_ts)

    # TODO start - inputchunglength to get an appropriate series to use for predicting

    model = TorchForecastingModel.load(model_folder + '/model.pkl')
    logging.info(f'Loaded successful model of class {type(model)}')
    if model.supports_future_covariates and model.supports_past_covariates:
        logging.info("Running prediction with full covariates")
        prediction = model.predict(
            n=96,  # TODO end_date - start_date days hours * 4 for PTUs
            series=sm_tss[0],
            past_covariates=wa_ts,
            future_covariates=wf_ts
        )
    elif model.supports_future_covariates:
        logging.info("Running prediction with future covariates")
        prediction = model.predict(
            n=96,  # TODO end_date - start_date days hours * 4 for PTUs
            series=sm_tss[0],
            future_covariates=wf_ts
        )
    else:
        logging.info("Running prediction with no covariates")
        prediction = model.predict(
            n=96,  # TODO end_date - start_date days hours * 4 for PTUs
            series=sm_tss[0]
        )
    print(prediction)
    # evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the prediction of the given timeframe using a pre-trained model '
                    'and evaluate its performance and accuracy')
    parser.add_argument('-smd', '--smart-meter-data', metavar='SMD_FILES', type=str, nargs='+',
                        help='comma-separated list of smart meter data csv files to be used for training')
    parser.add_argument('-wfd', '--weather-forecast-data', metavar='WFD_FILES', type=str,
                        help='comma-separated list of weather forecast csv files to be used for training')
    parser.add_argument('-wad', '--weather-actuals-data', metavar='WAD_FILES', type=str,
                        help='comma-separated list of weather actuals csv files to be used for training')
    parser.add_argument('-md', '--model-configuration', metavar='MODEL_CONFIG_PATH', type=str,
                        help='path to the model configuration YAML file')
    parser.add_argument('-sv', '--save-model-as', metavar='MODEL_SAVE_PATH', type=str,
                        help='path where model should be saved')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    smd_files = []
    if args.smart_meter_data:
        smd_files = [file.strip() for file in ','.join(args.smart_meter_data).split(',')]

    with open(args.model_configuration, 'r') as file:
        logging.info(f'Loading config from {args.model_configuration}')
        model_config = yaml.safe_load(file)

    # TODO make args
    start_date = dt.datetime(2023, 7, 1, 0, 0, tzinfo=AMS_TZ)
    end_date = dt.datetime(2023, 7, 8, 0, 0, tzinfo=AMS_TZ)

    main_evaluate(_config=model_config, start_date_time=start_date, end_date_time=end_date,
                  smart_meter_data_files=smd_files, weather_forecast_file=args.weather_forecast_data,
                  weather_actuals_file=args.weather_actuals_data,
                  model_folder=args.save_model_as)
