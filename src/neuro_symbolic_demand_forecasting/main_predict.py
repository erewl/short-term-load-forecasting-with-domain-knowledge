import logging
import datetime as dt

import argparse
import pickle

import pandas as pd
import yaml
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from neuro_symbolic_demand_forecasting.main_train import load_csvs, create_timeseries_from_dataframes, AMS_TZ


def main_evaluate(_config: dict, start_date: str, end_date: str,
                  smart_meter_data_files: list[str],
                  weather_forecast_file: str, weather_actuals_file: str, model_folder: str):
    sms, wf, wa = load_csvs(_config, smart_meter_data_files, [weather_forecast_file], [weather_actuals_file])
    end_date_dt = pd.to_datetime(end_date).tz_localize(AMS_TZ)
    actuals = sms[0][end_date_dt:end_date_dt + pd.Timedelta(days=7)]

    sms = [s[:end_date] for s in sms]

    sm_tss, wf_ts, wa_ts = create_timeseries_from_dataframes(sms, wf, wa, scale=False)

    # initializing unfitted scalers that are getting loaded from a pickle in this step
    with open(model_folder + '/scalers.pkl', 'rb') as f:
        non_pv_scaler = pickle.load(f)
        pv_scaler = pickle.load(f)
        wa_scaler = pickle.load(f)
        wf_scaler = pickle.load(f)

    pv_tss = [pv_scaler.transform(s) for s in sm_tss if any(s.min(axis=0) < 0)]
    non_pv_tss = [non_pv_scaler.transform(s) for s in sm_tss if any(s.min(axis=0) >= 0)]

    scaler = pv_scaler
    if len(pv_tss) < 0:
        logging.info("using non_pv_scaler")
        scaler = non_pv_scaler
    else:
        logging.info("Using pv_scaler")
    wf_ts: TimeSeries = wf_scaler.transform(wf_ts)
    wf_ts.pd_dataframe()
    wa_ts = wa_scaler.transform(wa_ts)
    sm_tss = pv_tss + non_pv_tss
    sm_tss = sm_tss[0] if len(sm_tss) == 1 else sm_tss

    model = TorchForecastingModel.load(model_folder + '/train_model.pkl')
    logging.info(f'Loaded successful model of class {type(model)}')
    if model.supports_future_covariates and model.supports_past_covariates:
        logging.info("Running prediction with full covariates")
        prediction = model.predict(
            n=96,
            series=sm_tss,
            past_covariates=wa_ts,
            future_covariates=wf_ts
        )
    elif model.supports_future_covariates:
        logging.info("Running prediction with future covariates")
        prediction = model.predict(
            n=672,  # TODO end_date - start_date days hours * 4 for PTUs
            series=sm_tss,
            future_covariates=wf_ts
        )
    else:
        logging.info("Running prediction with no covariates")
        prediction = model.predict(
            n=96,  # TODO end_date - start_date days hours * 4 for PTUs
            series=sm_tss
        )

    rescaled_prediction = scaler.inverse_transform(prediction)

    print(rescaled_prediction)
    with open(model_folder + '/val_results_pv_larger_unscaled.pkl', 'wb') as f:
        pickle.dump(prediction, f)
        pickle.dump(actuals, f)
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
    # start_date = dt.datetime(2023, 7, 8, 0, 0, tzinfo=AMS_TZ)
    # end_date = dt.datetime(2023, 7, 14, 23, 45, tzinfo=AMS_TZ)

    main_evaluate(_config=model_config, start_date='2023-07-15 00:00', end_date='2023-07-22 23:45',
                  smart_meter_data_files=smd_files, weather_forecast_file=args.weather_forecast_data,
                  weather_actuals_file=args.weather_actuals_data,
                  model_folder=args.save_model_as)
