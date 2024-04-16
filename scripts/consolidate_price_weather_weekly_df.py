import numpy as np

from scripts.load_data import load_price_df, load_weather_weekly_df
import pandas as pd

price_df = load_price_df('../processed_data/prices-2016-2024.csv')
weather_df = load_weather_weekly_df('../processed_data/weather_2016_2023.csv')
price_weather_weekly_df = pd.merge(price_df, weather_df, on='first_day_week', how='left')
price_weather_weekly_df.index = price_df.index

with open('../processed_data/price_weather_weekly_df.csv', 'w') as f:
    price_weather_weekly_df.to_csv(f)
