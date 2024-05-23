import logging
import datetime as dt

import argparse
import pickle

from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from neuro_symbolic_demand_forecasting.main_train import load_csvs, create_timeseries_from_dataframes


def main_evaluate(_config: dict, start_date_time: dt.datetime, end_date_time: dt.datetime,
                  smart_meter_data_files: list[str],
                  weather_forecast_file: str, weather_actuals_file: str, model_folder: str):
    pass
    sms, wf, wa = load_csvs(_config, smart_meter_data_files, [weather_forecast_file], [weather_actuals_file])
    wf = wf[0]
    wa = wa[0]

    sm_tss, wf_ts, wa_ts = create_timeseries_from_dataframes(sms, wf, wa, scale=False)

    # initializing unfitted scalers that are getting loaded from a pickle in this step
    pv_scaler, non_pv_scaler, wa_scaler, wf_scaler = Scaler(), Scaler(), Scaler(), Scaler()
    with open(model_folder + '/scaler.pkl') as f:
        pv_scaler, non_pv_scaler, wa_scaler, wf_scaler = pickle.load(f)

    sm_tss = [pv_scaler.transform(s) for s in sm_tss]
    wf_ts = wf_scaler.transform(wf_ts)
    wa_ts = wf_scaler.transform(wa_ts)

    with open(model_folder + '/model.pkl') as f:
        # TODO how to do custom loss here... (also pickle that?)
        model: TorchForecastingModel = pickle.load(f)

        prediction = []
        if model.supports_future_covariates and model.supports_past_covariates:
            prediction = model.predict(
                n=96,  # TODO end_date - start_date days hours * 4 for PTUs
                series=sm_tss[0],
                past_covariates=wa_ts,
                future_covariates=wf_ts
            )
        elif model.supports_future_covariates:
            prediction = model.predict(
                n=96,  # TODO end_date - start_date days hours * 4 for PTUs
                series=sm_tss[0],
                future_covariates=wf_ts
            )
        else:
            prediction = model.predict(
                n=96,  # TODO end_date - start_date days hours * 4 for PTUs
                series=sm_tss[0]
            )

        # evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the prediction of the given timeframe using a pre-trained model '
                    'and evaluate its performance and accuracy')
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

    if args.smart_meter_data:
        smd_files = [file.strip() for file in ','.join(args.smart_meter_data).split(',')]
    if args.weather_forecast_data:
        wfd_files = [file.strip() for file in ','.join(args.weather_forecast_data).split(',')]
    if args.weather_actuals_data:
        wad_files = [file.strip() for file in ','.join(args.weather_actuals_data).split(',')]

    main_predict()
