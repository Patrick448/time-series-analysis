import math
import sys
import numpy as np  # linear algebra
from scipy.stats import randint
import pandas as pd
from datetime import timedelta


def load_weather_df(path):
    weather_df = pd.read_csv(path, index_col=0)
    weather_df.index = pd.to_datetime(weather_df.index, utc=True)
    weather_df.drop(columns=['x'], inplace=True)
    weather_df.replace(-9999, np.NAN, inplace=True)
    weather_df = weather_df.interpolate(method='linear')

    weather_df = weather_df[(weather_df.index >= '2016-04-22') & (weather_df.index <= '2023-12-31')]

    return weather_df


def load_weather_weekly_df(path):
    weather_df = load_weather_df(path)

    weather_weekly_df = pd.DataFrame(
         weather_df.resample('W').mean())

    weather_weekly_df["first_day_week"] = (
               weather_weekly_df.index - weather_weekly_df.index.weekday * timedelta(days=1))

    return weather_weekly_df


def load_price_df(path):
    df = pd.read_csv(path, index_col='dt')
    df.index = pd.to_datetime(df.index, utc=True)

    df = df[(df.index >= '2016-04-22') & (df.index <= '2023-12-31')]

    resampled_df = df.resample('W').mean()
    resampled_df = resampled_df.interpolate(method='linear')
    #resampled_df.index = resampled_df.index.tz_localize("UTC")
    resampled_df.sort_values(by=['dt'], inplace=True)
    resampled_df[
        "first_day_week"] = resampled_df.index - resampled_df.index.weekday * timedelta(
        days=1)

    return resampled_df


