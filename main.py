import json
import os
import shutil
from os.path import abspath

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from models.ModelTraining import ModelTraining
import pandas as pd
import argparse

# Create the parser
arg_parser = argparse.ArgumentParser(description='Run LSTM model experiment')
# Add the arguments

arg_parser.add_argument('-config-file', '-cf',
                        type=str,
                        help='path to the JSON config file')


arg_parser.add_argument('-output-header', '-oh',
                        action='store_true',
                        help='print the header to the output file with the results')

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
results_path = None
model_params = None
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
        train_valid_test = config.get('train_valid_test')
        results_path = config.get('results_path')
        model_params = config.get('model_params')

results_path = f"res_{args.config_file}"

df = pd.read_csv(input_file, index_col=0)


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
    #os.mkdir(model_path)

def save_pred_ref(ref_pred_path, pred, ref,  model_id, dt=None, extra=None):

    pred_df =pd.DataFrame(pred)
    ref_df =pd.DataFrame(ref)

    if dt is not None:
        pred_df.index = dt
        ref_df.index= dt

    if extra is not None:
        pred_df['extra'] = extra

    pred_df.to_csv(f'{ref_pred_path}/pred.csv')
    ref_df.to_csv(f'{ref_pred_path}/ref.csv')

def run_crossv(data, cols, in_size, out_size, keep_only, architecture, save_path=None, model_id=None,
               start_offset=None, end_offset=None, train_valid_test: tuple = None, results_path=None,
               model_params=None):
    model = ModelTraining()
    if len(cols) > 1:
        exog_cols = cols[1:]

        for col in exog_cols:
            data[col] = data[col].shift(out_size)
            data = data[out_size:]

    if results_path is None:
        results_path = f"pred_ref_{architecture}_{model_id}"
    run_model = None
    if architecture == 'simple_lstm_v0' or architecture == 'dia_lstm_v0':
        run_model = model.lstm_train_predict
    elif architecture == 'sarimax':
        run_model = model.sarimax_train_predict
    elif architecture == 'prophet':
        run_model = model.prophet_train_predict

    # separação dos dados
    tscv_split = TimeSeriesSplit(test_size=out_size, n_splits=10)
    pred_list = []
    ref_list = []
    # models = []

    for i_split, (train_index, test_index) in enumerate(tscv_split.split(data)):
        cv_train, cv_test = data.iloc[train_index], data.iloc[test_index]
        # cv_test = pd.concat([cv_train.tail(in_size), cv_test], axis="rows")
        test_Y, yhat = run_model(cv_train, cv_test, i_split, cols, in_size, out_size,
                                 keep_only, architecture, save_path, model_params=model_params)

        pred_list_local = [str(yhat.index[0])]
        pred_list_local.extend(yhat.values)
        pred_list.append(pred_list_local)

        ref_list_local = [str(test_Y.index[0])]
        ref_list_local.extend(test_Y.values)
        ref_list.append(ref_list_local)

    pred_df = pd.DataFrame(pred_list)
    pred_df.rename(columns={0: 'dt'}, inplace=True)
    ref_df = pd.DataFrame(ref_list)
    ref_df.rename(columns={0: 'dt'}, inplace=True)
    pred_df.set_index('dt', inplace=True)
    ref_df.set_index('dt', inplace=True)

    save_pred_ref(f"{results_path}", pred_df, ref_df, model_id)

try:
    os.mkdir(results_path)
    shutil.copyfile(args.config_file, results_path + '/config.json')
except Exception as e:
    raise Exception (f"It was not possible to create the results directory.Exception: {e}")
    exit(1)

run_crossv(
    df,
    columns,
    in_size,
    out_size,
    keep_only,
    architecture=model_name,
    save_path=model_path,
    model_id=model_id,
    start_offset=start_offset,
    end_offset=end_offset,
    train_valid_test=train_valid_test,
    results_path = results_path,
    model_params=model_params)