 TFT With encoders:
 
```commandline
    encoders = {
        'datetime_attribute': {'future': ['year', 'dayofweek', 'day', 'month'],
                               'past': ['dayofweek', 'day', 'month', 'year']},
        'custom': {'past': [encode_ptu], 'future': [encode_ptu]},
        # 'transformer': Scaler(),
        'tz': 'Europe/Amsterdam'
    }
```

- 0_0 -> smart_meter_series (input)
- 1_0 -> wind_speed
- 1_1 -> global_radiation
- 1_2 -> air_pressure
- 1_3 -> air_temperature
- 1_4 -> relative_humidity
- 1_5 -> day_of_week
- 1_6 -> day
- 1_7 -> month
- 1_8 -> year
- 1_9 -> ptu 
- 
- 2_x -> same things but for past future_covariates
- 3_x -> same things but for future_covariates

---

LSTM with encoders:
```
        encoders = {
            'datetime_attribute': {'future': ['dayofweek', 'day', 'month', 'year'],
                                   },
            'custom': {
                'future': [encode_ptu, part_of_day_encoder.encode]
            },
            # 'position': {'past': ['relative'], 'future': ['relative']},
            # 'transformer': Scaler(),
            'tz': 'Europe/Amsterdam'
        }
```

- 0_0 -> smart_meter_series (input)
- 1_0 -> wind_speed
- 1_1 -> global_radiation
- 1_2 -> air_pressure
- 1_3 -> air_temperature
- 1_4 -> relative_humidity
- 1_5 -> day_of_week
- 1_6 -> ptu
- 1_7 -> part_of_day
- 
- 2_x -> same things but for past future_covariates