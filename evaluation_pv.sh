#! /bin/bash

#FOLDER="01_results/20240610_1038_tft_semantic_0010"
CONFIG_PATH="configs/tft_baseline_finetuned_v1.yaml"

FOLDER=$1
echo "$FOLDER"
# PV

#echo "Running training evaluation"
## PV Train Eval
#poetry run python  -m src.neuro_symbolic_demand_forecasting.main_predict \
#-smd data/2023_05_cleaned_pv.csv \
#-wad data/2023-04_to_08-amsterdam-actuals_filled_gaps.csv \
#-wfd data/2023_weather_data_06_run_summer_from_04_to_08.csv \
#-md $CONFIG_PATH \
#-mf "$FOLDER" \
#-sd 2023-05-09 -ed 2023-05-24
#
#poetry run python  -m src.neuro_symbolic_demand_forecasting.main_predict \
#-smd data/2023_06_cleaned_pv.csv \
#-wad data/2023-04_to_08-amsterdam-actuals_filled_gaps.csv \
#-wfd data/2023_weather_data_06_run_summer_from_04_to_08.csv \
#-md $CONFIG_PATH \
#-mf "$FOLDER" \
#-sd 2023-06-09 -ed 2023-06-23
#
#echo ""
#echo "Running validation evaluation..."
## PV Validation Eval
#poetry run python  -m src.neuro_symbolic_demand_forecasting.main_predict \
#-smd data/2023_05_cleaned_pv.csv \
#-wad data/2023-04_to_08-amsterdam-actuals_filled_gaps.csv \
#-wfd data/2023_weather_data_06_run_summer_from_04_to_08.csv \
#-md $CONFIG_PATH \
#-mf "$FOLDER" \
#-sd 2023-05-24 -ed 2023-05-31
#
#poetry run python  -m src.neuro_symbolic_demand_forecasting.main_predict \
#-smd data/2023_06_cleaned_pv.csv \
#-wad data/2023-04_to_08-amsterdam-actuals_filled_gaps.csv \
#-wfd data/2023_weather_data_06_run_summer_from_04_to_08.csv \
#-md $CONFIG_PATH \
#-mf "$FOLDER" \
#-sd 2023-06-23 -ed 2023-06-30

echo ""
echo "Running test evaluation"
# PV Test Eval
poetry run python  -m src.neuro_symbolic_demand_forecasting.main_predict \
-smd data/2023_07_cleaned_pv_rework.csv \
-wad data/2023-04_to_08-amsterdam-actuals_filled_gaps.csv \
-wfd data/2023_weather_data_06_run_summer_from_04_to_08.csv \
-md $CONFIG_PATH \
-mf "$FOLDER" \
-sd 2023-07-08 -ed 2023-07-31
