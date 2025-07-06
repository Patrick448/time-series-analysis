import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

original_df = pd.read_csv('../../processed_data/price_weather_weekly_df-2016-2023.csv', parse_dates=['dt'], index_col='dt')

def sarimax(ts,exo,all_param):
    results = []
    for param in all_param:
        try:
            mod = SARIMAX(ts,
                          exog = exo,
                          order=param[0],
                          seasonal_order=param[1])
            res = mod.fit()
            results.append((res,res.aic,param))
            print('Tried out SARIMAX{}x{} - AIC:{}'.format(param[0], param[1], round(res.aic,2)))
        except Exception as e:
            print(e)
            continue

    return results
# set parameter range
p,d,q = range(0,2),[1],range(0,2)
P,D,Q,s = range(0,2),[1],range(0,2),[56]
# list of all parameter combos
pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(P, D, Q, s))
all_param = list(itertools.product(pdq,seasonal_pdq))
len(all_param)

train = original_df['Alface Crespa - Roça']
exo_train = None
all_res = sarimax(train,exo_train,all_param)