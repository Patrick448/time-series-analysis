import pandas as pd
import re

df = pd.read_csv('../input/20240331174102-precos-medios.csv', parse_dates={'dt' : ['Ano', 'Mês', 'Dia']}, index_col='dt')
df.index = df.index.tz_localize('UTC')
df.drop(columns=['Região', 'Moeda'], inplace=True)
df['Unidade'] = [int(re.findall('\d+',i)[0]) for i in df['Unidade']]
df['Preço'] = [float(re.sub(',', '.', i)) for i in df['Preço']]
df.columns = ['produto', 'unidade', 'preco']
df['preco_unitario'] = df['preco'] / df['unidade']

products = df['produto'].unique()
new_df = pd.DataFrame()
new_df.index = df.index.drop_duplicates()
for product in products:
    temp_df = df[df['produto'] == product][['preco_unitario']]
    temp_df.columns = [product]
    new_df =new_df.join(temp_df, how='outer')

new_df.to_csv('../processed_data/prices-2016-2024.csv')