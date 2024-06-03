import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

#parse args
parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, required=True)
parser.add_argument('--output_path','-o', type=str, required=False)
parser.add_argument('--create_dir', '-cd', action='store_true', required=False)
args = parser.parse_args()


def plot_error_by_experiment_group(path, column_name, title, output_path="./"):
    df = pd.read_csv(path)
    ids = df['experiment_group'].unique().tolist()

    for i in ids:
        df_i = df[df['experiment_group'] == i]
        string_list = df_i[column_name].tolist()
        list_split = [x.split(',') for x in string_list]
        val_list = [[float(x) for x in y] for y in list_split]
        val_list = np.array(val_list)
        plt.plot(val_list.mean(axis=0), label=i, marker='o')
        plt.title(title)
        plt.legend()



    plt.savefig(f"{output_path}/{column_name}.png", bbox_inches='tight')
    plt.close()
    #plt.show()


def main():
    columns = ['RMSE_by_timestep', 'MAE_by_timestep', 'MAPE_by_timestep', 'MSE_by_timestep', 'R2_by_timestep']
    titles = ['RMSE by timestep', 'MAE by timestep', 'MAPE by timestep', 'MSE by timestep', 'R2 by timestep']
    abs_input = os.path.abspath(args.path)
    print(f"Reading data from {abs_input}")
    abs_output = os.path.abspath(args.output_path)

    if args.create_dir:
        os.makedirs(args.output_path, exist_ok=True)

    for i in range(len(columns)):
        plot_error_by_experiment_group(abs_input, columns[i], titles[i], output_path=abs_output)

    print(f"Saved plots to {abs_output}")

if __name__ == '__main__':
    main()