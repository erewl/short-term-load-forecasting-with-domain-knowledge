import polars as pl


def filter_out_missing_values(_df: pl.LazyFrame, min_ptus: int = 2884) -> pl.LazyFrame:
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
        cols = ['gross']
    return _df.select(pl.col(on), [pl.col(col) for col in cols]) \
        .group_by(on) \
        .agg([pl.sum(col) for col in cols]) \
        .sort(on)


def clean_data(file_name: str) -> pl.LazyFrame:
    df = pl.read_csv(file_name, columns=['EAN_SHA256', 'LDN', 'ODN', 'READINGDATE'],
                     try_parse_dates=True,
                     new_columns=[f.lower() for f in ['EAN_SHA256', 'LDN', 'ODN', 'READINGDATE']])

    min_date, max_date = df['readingdate'].min(), df['readingdate'].max()
    min_ptus = get_date_range_in_ptus(min_date, max_date)
    print('Filtering out', max_date, min_date, min_ptus)

    return accumulate(generate_interval_diffs(filter_out_missing_values(df.lazy(), min_ptus)))
