import json
import os

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
arg_parser.add_argument('-result_file','-rf',
                        type=str,
                        help='path to save the results')
arg_parser.add_argument('-output-header','-oh',
                        action='store_true',
                        help='print the header to the output file with the results')
arg_parser.add_argument('-columns','-c',
                        type=str,
                        help='columns to use in the model separated by comma. Ex: "column1,column2"')

arg_parser.add_argument('-save_path','-sp',
                        type=str,
                        help='path to save the model')

args = arg_parser.parse_args()



if args.config_file:
    with open(args.config_file, 'r') as f:
        config = json.load(f)
        in_size = config['input_size']
        out_size = config['output_size']
        keep_only = config['keep_only']
        columns = config['columns']
        result_file = config['result_file']
        input_file = config['input_file']
        save_path = config['save_path']
        model_name = config['model']
else:
    columns = args.columns.split(';')
    in_size = args.in_size
    out_size = args.out_size
    keep_only = args.keep_only
    result_file = args.result_file
    output_header = args.output_header
    input_file = args.input_file
    save_path = args.save_path
    model_name = args.model


df = pd.read_csv(input_file, index_col=0)
model = LSTM1()


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
    save_path=model_path)

rmse = model.rmse
mae = model.mae
rmse_by_timestep = str(model.rmse_by_timestep['RMSE'].tolist()).strip('[]').replace(" ", "")
mae_by_timestep = str(model.mae_by_timestep['MAE'].tolist()).strip('[]').replace(" ", "")
loss = str(model.history['loss']).strip('[]').replace(" ", "")
val_loss = str(model.history['val_loss']).strip('[]').replace(" ", "")
columns_str = "\""+','.join(columns)+"\""
# Save the results
csv_string = ""
if args.output_header:
    csv_string = "model,save_path,in_size,out_size,keep_only,RMSE,MAE,RMSE_by_timestep,MAE_by_timestep,loss,val_loss,columns\n"
csv_string += f"{model_name},{model_path},{in_size},{out_size},{keep_only},{rmse},{mae},\"{rmse_by_timestep}\",\"{mae_by_timestep}\",\"{loss}\",\"{val_loss}\",{columns_str}\n"

if result_file:
    with open(result_file, 'a') as f:
        f.write(csv_string)

print(csv_string)
