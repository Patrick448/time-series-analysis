import numpy as np

from models.LSTM1 import LSTM1
from scripts.load_data import load_price_df, load_weather_df, load_weather_weekly_df
import pandas as pd
import argparse


# Create the parser
arg_parser = argparse.ArgumentParser(description='Run LSTM model')
# Add the arguments
arg_parser.add_argument('-in_size', '-is',
                        type=int,
                        help='input size')

arg_parser.add_argument('-out_size', '-os',
                        type=int,
                        help='output size')
arg_parser.add_argument('-keep_only_size', '-kos',
                        type=int,
                        help='number of timesteps to keep')
arg_parser.add_argument('-result_file','-rf',
                        type=str,
                        help='path to save the results')
arg_parser.add_argument('-output-header','-oh',
                        action='store_true',
                        help='print the header to the output file with the results')

args = arg_parser.parse_args()

price_df = load_price_df('processed_data/prices-2016-2024.csv')[['Alface Crespa - Roça', 'first_day_week']]
weather_weekly_df = load_weather_weekly_df('processed_data/weather_2016_2023.csv')
price_weather_df = pd.merge(price_df, weather_weekly_df, on='first_day_week', how='left')
price_weather_df.index = price_df.index

model = LSTM1()
model.run(
    price_weather_df,
    ['Alface Crespa - Roça', 'TEMPERATURA DO PONTO DE ORVALHO (°C)'],
    8,
    8,
    8)
rmse = model.rmse
rmse_by_timestep = str(model.rmse_by_timestep['RMSE'].tolist()).strip('[]')
loss = str(model.history['loss']).strip('[]')
val_loss = str(model.history['val_loss']).strip('[]')

# Save the results
csv_string = ""
if args.output_header:
    csv_string = "in_size,out_size,keep_only_size,RMSE,RMSE_by_timestep,loss,val_loss\n"
csv_string += f"{args.in_size},{args.out_size},{args.keep_only_size},{rmse},\"{rmse_by_timestep}\",\"{loss}\",\"{val_loss}\"\n"

if args.result_file:
    with open(args.result_file, 'a') as f:
        f.write(csv_string)

print(csv_string)
