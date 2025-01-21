import logging
import datetime as dt

import argparse
import pickle
from typing import Optional, List

import numpy as np
import pandas as pd
import pytz
import torch
import yaml
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.metrics.metrics import mae, mape, rmse, smape, r2_score, merr

from neuro_symbolic_demand_forecasting.main_train import load_csvs, create_timeseries_from_dataframes


def main_evaluate(_config: dict, start_date: str, end_date: str,
                  smart_meter_data_files: str,
                  weather_forecast_file: str, weather_actuals_file: str, model_folders: List[str]):
    # start and end should be within the range of the smartmeter data file
    # calculate metrics per day, but also for the entire timerange and write to file.
    for model_folder in model_folders:
        model = TorchForecastingModel.load(model_folder + '/train_model.pkl', map_location=torch.device('cpu'))
        model.to_cpu()
        logging.info(f'Loaded successful model of class {type(model)}')

        with open(f'{model_folder}/scalers.pkl', 'rb') as f:
            non_pv_scaler = pickle.load(f)
            pv_scaler = pickle.load(f)

            weather_actuals_scaler = pickle.load(f)
            weather_forecast_scaler = pickle.load(f)
            logging.info("Loaded the scalers successfully")

        sms, wf, wa = load_csvs(_config, [smart_meter_data_files], [weather_forecast_file], [weather_actuals_file])
        sm_tss, wf_ts, wa_ts = create_timeseries_from_dataframes(sms, wf, wa,
                                                                 add_static_covariates=model.uses_static_covariates,
                                                                 scale=False)
        assert len(sm_tss) == 1
        sm_tss = sm_tss[0]

        if (sm_tss.min(axis=0) < 0).values.any():
            scaler = pv_scaler
            prefix = 'pv'
        else:
            scaler = non_pv_scaler
            prefix = 'non_pv'

        scaled_smart_meter_ts = scaler.fit_transform(sm_tss)
        scaled_weather_actuals = weather_actuals_scaler.transform(wa_ts)
        scaled_weather_forecasts = weather_forecast_scaler.transform(wf_ts)

        scaled_inputs = []
        scaled_actualss = []
        dates = []
        for d in pd.date_range(start_date, end_date):  # could also be pass as an array of timeseries to the model.
            input_start, input_end = d - dt.timedelta(days=7), d - dt.timedelta(minutes=15)
            scaled_input = scaled_smart_meter_ts[input_start:input_end]
            forecast_start, forecast_end = d, d + dt.timedelta(days=1) - dt.timedelta(minutes=15)
            logging.info(
                f"Using {input_start} to {input_end} to forecast {forecast_start} to {forecast_end}, Length: {len(scaled_input)}")
            scaled_actuals = scaled_smart_meter_ts[forecast_start:forecast_end]

            dates.append(d)
            scaled_inputs.append(scaled_input)
            scaled_actualss.append(scaled_actuals)

        scaled_predictions = model.predict(
            n=model.output_chunk_length,
            series=scaled_inputs,
            past_covariates=scaled_weather_actuals * len(scaled_inputs),
            future_covariates=scaled_weather_forecasts * len(scaled_inputs),
            n_jobs=5,
            num_loader_workers=5
        )

        rescaled_predictions = [scaler.inverse_transform(sp).with_columns_renamed(
            'gross', 'gross_predicted') for sp in scaled_predictions]
        rescaled_actuals = [scaler.inverse_transform(sa).with_columns_renamed(
            'gross', 'gross_actuals') for sa in scaled_actualss]

        # subset_prices = prices[rescaled_predictions.time_index.min():rescaled_predictions.time_index.max()]

        metrics = pd.DataFrame()
        ts: Optional[TimeSeries] = None

        for d, predicted, actuals in zip(dates, rescaled_predictions, rescaled_actuals):
            day_metrics = get_metrics(actuals, predicted)
            day_metrics['date'] = d
            metrics = pd.concat([metrics, pd.DataFrame([day_metrics])])

            if ts is None:
                ts = predicted.stack(actuals)
            else:
                ts = ts.append(predicted.stack(actuals))

        # subset_prices = prices[ts.time_index.min():ts.time_index.max()]
        logging.info("Total metrics:")
        total_metrics = get_metrics(ts['gross_actuals'], ts['gross_predicted'])
        total_metrics['date'] = 'full'

        metrics = pd.concat([metrics, pd.DataFrame([total_metrics])])

        metrics.to_csv(f'{model_folder}/{end_date}_{start_date}_{prefix}_metrics.csv')
        ts.to_csv(f'{model_folder}/{end_date}_{start_date}_{prefix}_evaluation_data.csv')


def _compute_imbalance_result(_target, _predicted, _prices):
    # error = _predicted - _target # double check signs (when production is -, i might need to flip pred and target)
    error = _target - _predicted
    imbalance_short = _prices['IMBALANCE_SHORT_EUR_MWH'].values()
    imbalance_long = _prices['IMBALANCE_LONG_EUR_MWH'].values()
    spot_price = _prices['SPOT_EUR_MWH'].values()
    # maybe not use imbalance costs, but use imbalance MWh

    error_vals = error.values()
    ib_result = np.where(
        error_vals >= 0,
        # when predicted is lower than target -> we are short
        error_vals * (imbalance_short - spot_price) / 1000 / 1000,
        # when predicted is higher than target -> we are long
        error_vals * (imbalance_long - spot_price) / 1000 / 1000
    )
    return ib_result


def _compute_imbalance_volumes(_target, _predicted):
    # sum( | y - y(hat) |) / sum( | y |) *100(
    return np.sum(np.abs((_target - _predicted).values())) / \
           np.sum(np.abs(_target.values())) * 100


def smape_100(_target, _predicted):
    _t = _target.values()
    _p = _predicted.values()
    return 100 * np.mean(np.abs(_t - _p) /
                        (np.abs(_t) + np.abs(_p))
                        )


def get_metrics(_target, _predicted):
    smape_ = smape(_target, _predicted)
    smape_100_ = smape_100(_target, _predicted)
    # mape_ = mape(_target, _predicted)
    merr_ = merr(_target, _predicted)
    mae_ = mae(_target, _predicted)
    # wape_ = (_target - _predicted).abs().sum() / _target.sum()
    rmse_ = rmse(_target, _predicted)
    r2_ = r2_score(_target, _predicted)
    imbalance = _compute_imbalance_volumes(_target, _predicted)

    metrics = {
        'sMAPE': round(smape_, 3),
        'sMAPE_100': round(smape_100_, 3),
        'WAPE': round(imbalance, 3),
        'ME': round(merr_, 3),
        'MAE': round(mae_, 3),
        'RMSE': round(rmse_, 3),
        'R2': round(r2_, 3)
    }

    logging.info('\n'.join([f"{k}: {v} " for k, v in metrics.items()]))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the prediction of the given timeframe using a pre-trained model '
                    'and evaluate its performance and accuracy')
    parser.add_argument('-smd', '--smart-meter-data', metavar='SMD_FILES', type=str,
                        help='comma-separated list of smart meter data csv files to be used for training')
    parser.add_argument('-wfd', '--weather-forecast-data', metavar='WFD_FILES', type=str,
                        help='comma-separated list of weather forecast csv files to be used for training')
    parser.add_argument('-wad', '--weather-actuals-data', metavar='WAD_FILES', type=str,
                        help='comma-separated list of weather actuals csv files to be used for training')
    parser.add_argument('-md', '--model-configuration', metavar='MODEL_CONFIG_PATH', type=str,
                        help='path to the model configuration YAML file')
    parser.add_argument('-mf', '--model-folders', metavar='MODEL_FOLDER', type=str, nargs='+',
                        help='path where model should be saved')
    parser.add_argument('-sd', '--start-date', metavar='START_DATE', type=str,
                        help='start date of the evaluation timewindow')
    parser.add_argument('-ed', '--end-date', metavar='END_DATE', type=str,
                        help='end date of the evaluation time window, e.g. 2023-07-12')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.model_configuration, 'r') as file:
        logging.info(f'Loading config from {args.model_configuration}')
        model_config = yaml.safe_load(file)

    prices = pd.read_csv('data/2023-04_to_08_prices.csv', parse_dates=['DELIVERY_DATETIME']).set_index(
        'DELIVERY_DATETIME')
    prices.index = prices.index.tz_convert(pytz.timezone('Europe/Amsterdam')).tz_localize(None)
    prices = TimeSeries.from_dataframe(prices, freq='15min')

    mf = []
    if args.model_folders:
        mf = [file.strip() for file in ','.join(args.model_folders).split(',')]
    print(args)
    main_evaluate(_config=model_config, start_date=args.start_date, end_date=args.end_date,
                  smart_meter_data_files=args.smart_meter_data, weather_forecast_file=args.weather_forecast_data,
                  weather_actuals_file=args.weather_actuals_data,
                  model_folders=mf)
