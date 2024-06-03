import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from custom_transforms.transforms import *
from statsmodels.tsa.stattools import adfuller
from scripts.load_data import load_weather_df, load_price_df
from datetime import timedelta


def add_shifted_column(shift_amount, col_name, df, neg=True):
    added_cols = []
    for i in range(1, shift_amount + 1):
        _i = -i if neg else i
        df[f'{col_name}_{_i}'] = df[col_name].shift(_i).rolling(window=4).mean()
        added_cols.append(f'{col_name}_{_i}')

    return added_cols


def run():
    """
    This function runs several data transformation steps and
    consolidates the data into a single dataframe.
    """

    price_df = load_price_df('../processed_data/prices-2016-2024.csv')
    weather_df = load_weather_df('../processed_data/weather_2016_2023.csv')
    weather_df_resampled = weather_df.resample('W').mean()
    weather_df_resampled["first_day_week"] = (
                weather_df_resampled.index - weather_df_resampled.index.weekday * timedelta(days=1))
    price_weather_weekly_df = pd.merge(price_df, weather_df_resampled, on='first_day_week', how='left')
    price_weather_weekly_df.index = price_df.index
    price_weather_weekly_df = price_weather_weekly_df.drop(columns=['first_day_week'])
    weather_df_resampled = weather_df_resampled.drop(columns=['first_day_week'])

    for col in weather_df.columns:
        price_weather_weekly_df[f'{col}_mean'] = weather_df[col].resample('W').mean()  # .rolling(window=4).mean()
        price_weather_weekly_df[f'{col}_max'] = weather_df[col].resample('W').max()  # .rolling(window=4).mean()
        price_weather_weekly_df[f'{col}_sum'] = weather_df[col].resample('W').sum()  # .rolling(window=4).mean()
        price_weather_weekly_df[f'{col}_min'] = weather_df[col].resample('W').min()
        price_weather_weekly_df[f'{col}_std'] = weather_df[col].resample('W').std()
        price_weather_weekly_df[f'{col}_var'] = weather_df[col].resample('W').var()
        price_weather_weekly_df[f'{col}_median'] = weather_df[col].resample('W').median()
        price_weather_weekly_df[f'{col}_ma4'] = weather_df[col].rolling(window=4).mean()

        price_weather_weekly_df.drop(columns=[col], inplace=True)
        price_weather_weekly_df = price_weather_weekly_df.copy()


    #price_weather_weekly_df["Alface Crespa - Roça_+49"] = price_weather_weekly_df["Alface Crespa - Roça"].shift(49)
    #price_weather_weekly_df["Alface Crespa - Roça_+50"] = price_weather_weekly_df["Alface Crespa - Roça"].shift(50)
    #price_weather_weekly_df["Alface Crespa - Roça_+51"] = price_weather_weekly_df["Alface Crespa - Roça"].shift(51)
    #price_weather_weekly_df["Alface Crespa - Roça_+52"] = price_weather_weekly_df["Alface Crespa - Roça"].shift(52)
    #price_weather_weekly_df["Alface Crespa - Roça_+53"] = price_weather_weekly_df["Alface Crespa - Roça"].shift(53)

   # for col in price_weather_weekly_df.columns:
   #     for i in range(49, 53):
   #         price_weather_weekly_df[f"{col}_+{i}"] = price_weather_weekly_df[col].shift(i)
    #        price_weather_weekly_df = price_weather_weekly_df.copy()

    #interest_points = [i for i in range(0, 62, 26)]
    #for col in price_weather_weekly_df.columns:
    #    for x in interest_points:
    #        price_weather_weekly_df[f"{col}_+{x}"] = price_weather_weekly_df[col].shift(x)
    #        for i in range(7, 11):
    #            price_weather_weekly_df[f"{col}_+{x+i}"] = price_weather_weekly_df[col].shift(x+i)
    #            price_weather_weekly_df = price_weather_weekly_df.copy()


    #added_shifted_cols = add_shifted_column(12, "Alface Crespa - Roça", price_weather_weekly_df)

    price_weather_weekly_df = price_weather_weekly_df[price_weather_weekly_df.index >= '2017-04-30']
    price_weather_weekly_df.to_csv('../processed_data/price_weather_weekly_df.V2.csv')


if __name__ == '__main__':
    run()
