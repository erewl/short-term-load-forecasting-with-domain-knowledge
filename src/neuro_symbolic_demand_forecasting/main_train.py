import argparse
import logging
import yaml
from darts.models import TFTModel

from neuro_symbolic_demand_forecasting.darts.loss import CustomLoss


def main_train(smd_files: list[str], wfd_files: list[str], wad_files: list[str], model_config_path: str):
    # load_dotenv()

    # load data, smart meter, weather, actuals
    # transform to darts.Timeseries
    with open(model_config_path, 'r') as file:
        model_config = yaml.safe_load(file)
    model = init_model(model_config)
    # train
    # validate


def build_torch_kwargs(kwargs: dict)-> dict:
    if kwargs['loss_fn'] and kwargs['loss_fn']=='CustomLoss':
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
                **{k: v for k, v in tft_config.items() if k not in ['input_chunk_length', 'output_chunk_length', 'torch_kwargs']},
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

    logging.info('Starting training!')
    main_train(smd_files, wfd_files, wad_files, args.model_configuration)
