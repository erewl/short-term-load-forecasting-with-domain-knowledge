import polars as pl
import numpy as np
import logging


def filter_out_missing_values(_df: pl.LazyFrame, min_ptus: int = 2884) -> pl.LazyFrame:
    print(f"Filtering out timeseries with less than {min_ptus}")
    return _df.group_by('ean_sha256') \
        .map_groups(lambda group:
                    group.filter(min_ptus >= len(group)), schema=None) \
        .sort('readingdate')


def get_date_range_in_ptus(mi, ma):
    rnge = ma - mi
    ptus = (((rnge.days * 24 * 60) + 15 + (
            rnge.seconds / 60)) / 60 * 4)  # (days to hours to minute (+15 mins, since incl.) + seconds to minutes) to hours to ptus
    return ptus


def generate_interval_diffs(_df: pl.LazyFrame) -> pl.LazyFrame:
    print(f'Generating intervals')
    return _df.sort('readingdate').with_columns(
        (pl.col('ldn') - pl.col('ldn').shift(1)).over('ean_sha256').alias('ldn_diff'),
        (pl.col('odn') - pl.col('odn').shift(1)).over('ean_sha256').alias('odn_diff')
    ).with_columns(
        (pl.col('ldn_diff') - pl.col('odn_diff')).alias('gross')
    )


def accumulate(_df: pl.LazyFrame, on: str = 'readingdate', cols=None) -> pl.LazyFrame:
    """
    aggregating dataframe by grouping by timestamp and summing values up
    """
    if cols is None:
        cols = 'gross'
    print(f'Aggregating data into one timeseries: cols: {cols}')
    return _df.select(pl.col(on), pl.col(cols)) \
        .group_by(on) \
        .agg(pl.sum(cols)) \
        .sort(on)


def detect_outliers(_df: pl.DataFrame, col: str = 'gross') -> list:
    q3, q1 = np.percentile(_df[[col]], [75, 25])
    iqr = q3 - q1
    # fine tune the multiplicator based on the visual
    upper_bound = q3 + (0.8 * iqr)
    lower_bound = q1 - (1.5 * iqr)

    print(f'Checking for outliers above {upper_bound} and below {lower_bound}')
    return _df.with_row_index().filter((pl.col(col) > upper_bound) | (pl.col(col) < lower_bound))['index']


def interpolate_outliers(_df: pl.DataFrame, outlier_indices: list, col: str = 'gross') -> pl.DataFrame:
    """
    Linear interpolates outliers, returns fixed dataframe
    """
    print(f'Linearly interpolation outliers detected at the indices: {outlier_indices}')
    for idx in outlier_indices:
        # Get the previous and next values
        prev_value = _df[col][idx - 1]
        next_value = _df[col][idx + 1]

        # Linear interpolation
        interpolated_value = (prev_value + next_value) / 2
        _df[idx, 'gross'] = interpolated_value
    return _df


def clean_data(_df: pl.DataFrame) -> pl.DataFrame:
    min_date, max_date = _df['readingdate'].min(), _df['readingdate'].max()
    min_ptus = get_date_range_in_ptus(min_date, max_date)
    print('Filtering out', max_date, min_date, min_ptus)

    con =  len(_df['ean_sha256'].unique())
    df = filter_out_missing_values(_df.lazy(), min_ptus)
    newcon = len(df.collect()['ean_sha256'].unique())
    print("Connections with missing values !", con, newcon, con-newcon)
    df = generate_interval_diffs(df)
    df = accumulate(df)
    print("Collecting results!")
    df = df.collect()
    outliers = detect_outliers(df)
    print(len(df), len(outliers))
    df = interpolate_outliers(df, outliers)
    return df
