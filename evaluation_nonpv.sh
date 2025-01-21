#! /bin/bash

#FOLDER="01_results/20240610_1038_tft_semantic_0010"
CONFIG_PATH="configs/tft_baseline_finetuned_v1.yaml"

FOLDER=$1
echo "$FOLDER"

# NonPV

echo "Running training evaluation"
# NonV Train Eval
#poetry run python  -m src.neuro_symbolic_demand_forecasting.main_predict \
#-smd data/2023_05_cleaned_non_pv.csv \
#-wad data/2023-04_to_08-amsterdam-actuals_filled_gaps.csv \
#-wfd data/2023_weather_data_06_run_summer_from_04_to_08.csv \
#-md $CONFIG_PATH \
#-mf "$FOLDER" \
#-sd 2023-05-09 -ed 2023-05-24

#poetry run python  -m src.neuro_symbolic_demand_forecasting.main_predict \
#-smd data/2023_06_cleaned_non_pv.csv \
#-wad data/2023-04_to_08-amsterdam-actuals_filled_gaps.csv \
#-wfd data/2023_weather_data_06_run_summer_from_04_to_08.csv \
#-md $CONFIG_PATH \
#-mf "$FOLDER" \
#-sd 2023-06-09 -ed 2023-06-23
#
#echo ""
#echo "Running validation evaluation..."
## NonPV Validation Eval
#poetry run python  -m src.neuro_symbolic_demand_forecasting.main_predict \
#-smd data/2023_05_cleaned_non_pv.csv \
#-wad data/2023-04_to_08-amsterdam-actuals_filled_gaps.csv \
#-wfd data/2023_weather_data_06_run_summer_from_04_to_08.csv \
#-md $CONFIG_PATH \
#-mf "$FOLDER" \
#-sd 2023-05-24 -ed 2023-05-31
#
#poetry run python  -m src.neuro_symbolic_demand_forecasting.main_predict \
#-smd data/2023_06_cleaned_non_pv.csv \
#-wad data/2023-04_to_08-amsterdam-actuals_filled_gaps.csv \
#-wfd data/2023_weather_data_06_run_summer_from_04_to_08.csv \
#-md $CONFIG_PATH \
#-mf "$FOLDER" \
#-sd 2023-06-23 -ed 2023-06-30
#
echo ""
echo "Running test evaluation"
# NonPV Test Eval
poetry run python  -m src.neuro_symbolic_demand_forecasting.main_predict \
-smd data/2023_07_cleaned_non_pv_rework.csv \
-wad data/2023-04_to_08-amsterdam-actuals_filled_gaps.csv \
-wfd data/2023_weather_data_06_run_summer_from_04_to_08.csv \
-md $CONFIG_PATH \
-mf "$FOLDER" \
-sd 2023-07-08 -ed 2023-07-31

cluster ,sMAPE,sMAPE_100,WAPE,ME,MAE,RMSE,R2

PreTuning PV,52.439,26.219,34.366,-59181.598,155827.66,225538.72,0.841
PreTuning NonPV,7.85,3.925,8.087,-3355.688,12830.873,16499.938,0.86

Baseline PV, 38.527,19.264,25.676,-37695.605,116422.375,185219.92,0.893
Baseline NonPV, 8.634,4.317,8.645,-10173.375,13715.661,16942.963,0.852
