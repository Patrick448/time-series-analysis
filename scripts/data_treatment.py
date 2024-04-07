from typing import Dict
import pandas as pd


def compute_missing_values_by_col(df: pd.DataFrame) -> (pd.DataFrame, Dict):
    missing_df_mask = pd.DataFrame()
    cols_with_missing_values = []
    missing_values_by_col_dict_list = []
    for col in df.columns:
        df_missing = df[[col]][(df[col].isnull()) | (df[col] == -9999)]
        missing_df_mask[col] = (df[col].isnull()) | (df[col] == -9999)
        if len(df_missing) > 0:
            cols_with_missing_values.append(col)
            missing_values_by_col_dict_list.append({'col': col, 'missing_values': len(df_missing), 'missing_values_ratio': len(df_missing) / len(df)})

    missing_values_by_col_df = pd.DataFrame(missing_values_by_col_dict_list)
    missing_block_sizes = {}

    for col in missing_values_by_col_df['col']:
        missing_block_sizes[col] = []
        values = missing_df_mask[col].values
        count = 0
        for v in values:
            if v:
                count += 1
            else:
                if count > 0:
                    missing_block_sizes[col].append(count)
                    count = 0

    return missing_values_by_col_df, missing_block_sizes

