import json
import os
from os.path import abspath

import numpy as np

from models.LSTM1 import LSTM1
import pandas as pd
import argparse

# Create the parser
arg_parser = argparse.ArgumentParser(description='Run LSTM model experiment')
# Add the arguments

arg_parser.add_argument('-input-file', '-if',
                        type=str,
                        help='path to the file containing the input data')

arg_parser.add_argument('-config-file', '-cf',
                        type=str,
                        help='path to the JSON config file')

arg_parser.add_argument('-model', '-m',
                        type=str,
                        help='name of the model to run')

arg_parser.add_argument('-in_size', '-is',
                        type=int,
                        help='input size')

arg_parser.add_argument('-out_size', '-os',
                        type=int,
                        help='output size')
arg_parser.add_argument('-keep_only', '-ko',
                        type=int,
                        help='number of timesteps to keep')
arg_parser.add_argument('-result_file', '-rf',
                        type=str,
                        help='path to save the results')
arg_parser.add_argument('-output-header', '-oh',
                        action='store_true',
                        help='print the header to the output file with the results')
arg_parser.add_argument('-columns', '-c',
                        type=str,
                        help='columns to use in the model separated by semicolon. Ex: "column1;column2"')

arg_parser.add_argument('-save_path', '-sp',
                        type=str,
                        help='path to save the model')
arg_parser.add_argument('-experiment_group', '-eg',
                        type=str,
                        help='name of the experiment group')
arg_parser.add_argument('-start_offset', '-so',
                        type=int,
                        help='start offset')
arg_parser.add_argument('-end_offset', '-eo',
                        type=int,
                        help='end offset')

args = arg_parser.parse_args()

columns = None
in_size = None
out_size = None
keep_only = None
result_file = None
output_header = None
input_file = None
save_path = None
model_name = None
experiment_group = None
start_offset = None
end_offset = None

if args.config_file:
    with open(args.config_file, 'r') as f:
        config = json.load(f)
        in_size = config.get('input_size')
        out_size = config.get('output_size')
        keep_only = config.get('keep_only')
        columns = config.get('columns')
        result_file = config.get('result_file')
        input_file = config.get('input_file')
        save_path = config.get('save_path')
        model_name = config.get('model')
        experiment_group = config.get('experiment_group')
        start_offset = config.get('start_offset')
        end_offset = config.get('end_offset')

columns = args.columns.split(';') if args.columns else columns
in_size = args.in_size if args.in_size else in_size
out_size = args.out_size if args.out_size else out_size
keep_only = args.keep_only if args.keep_only else keep_only
result_file = args.result_file if args.result_file else result_file
output_header = args.output_header if args.output_header else output_header
input_file = args.input_file if args.input_file else input_file
save_path = args.save_path if args.save_path else save_path
model_name = args.model if args.model else model_name
experiment_group = args.experiment_group if args.experiment_group else experiment_group
start_offset = args.start_offset if args.start_offset else start_offset
end_offset = args.end_offset if args.end_offset else end_offset


df = pd.read_csv(input_file, index_col=0)
model = LSTM1()

model_id = None
if save_path is not None:
    with open(f'{save_path}/last_id.txt', 'r') as f:
        last_id = int(f.read())
        model_id = last_id + 1
        f.close()
    with open(f'{save_path}/last_id.txt', 'w') as f:
        f.write(str(model_id))
        f.close()

    model_path = f'{save_path}/model_{model_name}.{model_id}'
    os.mkdir(model_path)

model.run(
    df,
    columns,
    in_size,
    out_size,
    keep_only,
    architecture=model_name,
    save_path=model_path,
    model_id=model_id,
    start_offset=start_offset,
    end_offset=end_offset)

rmse = model.rmse
mae = model.mae
mse = model.mse
mape = model.mape
r2 = model.r2

rmses = model.rmse_by_timestep['RMSE'].tolist()
maes = model.mae_by_timestep['MAE'].tolist()
mses = model.mse_by_timestep['MSE'].tolist()
mapes = model.mape_by_timestep['MAPE'].tolist()
r2s = model.r2_by_timestep['R2'].tolist()

rmse_by_timestep = "\"" + ",".join(map(str, rmses)) + "\""
mae_by_timestep = "\"" + ",".join(map(str, maes)) + "\""
mse_by_timestep = "\"" + ",".join(map(str, mses)) + "\""
mape_by_timestep = "\"" + ",".join(map(str, mapes)) + "\""
r2_by_timestep = "\"" + ",".join(map(str, r2s)) + "\""

loss = '\"'+','.join(map(str, model.history['loss'])) + '\"'
val_loss = '\"'+','.join(map(str, model.history['val_loss'])) + '\"'
columns_str = "\"" + ','.join(columns) + "\""

csv_columns = ['experiment_group', 'model','model_id', 'save_path', 'in_size', 'out_size', 'keep_only',
               'RMSE', 'MAE', 'MSE', 'MAPE', 'R2',
               'RMSE_by_timestep', 'MAE_by_timestep', 'MSE_by_timestep', 'MAPE_by_timestep', 'R2_by_timestep',
               'loss', 'val_loss', 'columns']

csv_values = [experiment_group, model_name, model_id, abspath(model_path), in_size, out_size, keep_only,
              rmse, mae, mse, mape, r2,
              rmse_by_timestep, mae_by_timestep, mse_by_timestep, mape_by_timestep, r2_by_timestep,
              loss, val_loss, columns_str]



# Save the results
csv_string = ""
if args.output_header:
    csv_string = ",".join(csv_columns) + "\n"
csv_string += ",".join(map(str, csv_values)) + "\n"

if result_file:
    with open(result_file, 'a') as f:
        f.write(csv_string)

print(csv_string)
