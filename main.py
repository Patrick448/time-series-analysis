import json

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
arg_parser.add_argument('-keep_only_size', '-kos',
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

args = arg_parser.parse_args()



if args.config_file:
    with open(args.config_file, 'r') as f:
        config = json.load(f)
        in_size = config['input_size']
        out_size = config['output_size']
        keep_only_size = config['keep_only_size']
        columns = config['columns']
        result_file = config['result_file']
        input_file = config['input_file']
else:
    columns = args.columns.split(';')
    in_size = args.in_size
    out_size = args.out_size
    keep_only_size = args.keep_only_size
    result_file = args.result_file
    output_header = args.output_header
    input_file = args.input_file


df = pd.read_csv(input_file, index_col=0)
model = LSTM1()

model.run(
    df,
    columns,
    in_size,
    out_size,
    keep_only_size)

rmse = model.rmse
rmse_by_timestep = str(model.rmse_by_timestep['RMSE'].tolist()).strip('[]').replace(" ", "")
loss = str(model.history['loss']).strip('[]').replace(" ", "")
val_loss = str(model.history['val_loss']).strip('[]').replace(" ", "")

# Save the results
csv_string = ""
if args.output_header:
    csv_string = "in_size,out_size,keep_only_size,RMSE,RMSE_by_timestep,loss,val_loss\n"
csv_string += f"{in_size},{out_size},{keep_only_size},{rmse},\"{rmse_by_timestep}\",\"{loss}\",\"{val_loss}\"\n"

if result_file:
    with open(result_file, 'a') as f:
        f.write(csv_string)

print(csv_string)
