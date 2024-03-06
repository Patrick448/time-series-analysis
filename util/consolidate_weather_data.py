import pandas as pd



def consolidate():
    columns = ["PRECIPITAÇÃO TOTAL, HORÁRIO (mm)","PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)","PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)","PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)","RADIACAO GLOBAL (Kj/m²)","TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)","TEMPERATURA DO PONTO DE ORVALHO (°C)","TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)","TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)","TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)","TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)","UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)","UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)","UMIDADE RELATIVA DO AR, HORARIA (%)","VENTO, DIREÇÃO HORARIA (gr) (° (gr))","VENTO, RAJADA MAXIMA (m/s)","VENTO, VELOCIDADE HORARIA (m/s)", "x"]

    weather_df_2023 = pd.read_csv('../input/INMET_SE_RJ_A618_TERESOPOLIS-PARQUE NACIONAL_01-01-2023_A_31-12-2023.CSV', sep=';',parse_dates={'dt': ['Data', 'Hora UTC']}, low_memory=False, decimal=',', na_values=['nan','?'], index_col='dt',skiprows=8, encoding='ISO-8859-1')
    weather_df_2023.columns = columns

    weather_df_2022 = pd.read_csv('../input/INMET_SE_RJ_A618_TERESOPOLIS-PARQUE NACIONAL_01-01-2022_A_31-12-2022.CSV', sep=';',parse_dates={'dt': ['Data', 'Hora UTC']}, low_memory=False, decimal=',', na_values=['nan','?'], index_col='dt',skiprows=8, encoding='ISO-8859-1')
    weather_df_2022.columns = columns

    weather_df_2021 = pd.read_csv('../input/INMET_SE_RJ_A618_TERESOPOLIS-PARQUE NACIONAL_01-01-2021_A_31-12-2021.CSV', sep=';',parse_dates={'dt': ['Data', 'Hora UTC']}, low_memory=False, decimal=',', na_values=['nan','?'], index_col='dt',skiprows=8, encoding='ISO-8859-1')
    weather_df_2021.columns = columns

    weather_df_2020 = pd.read_csv('../input/INMET_SE_RJ_A618_TERESOPOLIS-PARQUE NACIONAL_01-01-2020_A_31-12-2020.CSV', sep=';',parse_dates={'dt': ['Data', 'Hora UTC']}, low_memory=False, decimal=',', na_values=['nan','?'], index_col='dt',skiprows=8, encoding='ISO-8859-1')
    weather_df_2020.columns = columns

    weather_df_2019 = pd.read_csv('../input/INMET_SE_RJ_A618_TERESOPOLIS-PARQUE NACIONAL_01-01-2019_A_31-12-2019.CSV', sep=';',parse_dates={'dt': ['Data', 'Hora UTC']}, low_memory=False, decimal=',', na_values=['nan','?'], index_col='dt',skiprows=8, encoding='ISO-8859-1')
    weather_df_2019.columns = columns

    weather_df_2018 = pd.read_csv('../input/INMET_SE_RJ_A618_TERESOPOLIS_01-01-2018_A_31-12-2018.CSV', sep=';',parse_dates={'dt': ['DATA (YYYY-MM-DD)', 'HORA (UTC)']}, low_memory=False, decimal=',', na_values=['nan','?'], index_col='dt',skiprows=8, encoding='ISO-8859-1')
    weather_df_2018.columns = columns
    weather_df_2018.index = weather_df_2018.index.tz_localize("UTC").tz_convert("America/Sao_Paulo")


    weather_df_2017 = pd.read_csv('../input/INMET_SE_RJ_A618_TERESOPOLIS_01-01-2017_A_31-12-2017.CSV', sep=';',parse_dates={'dt': ['DATA (YYYY-MM-DD)', 'HORA (UTC)']}, low_memory=False, decimal=',', na_values=['nan','?'], index_col='dt',skiprows=8, encoding='ISO-8859-1')
    weather_df_2017.columns = columns
    weather_df_2017.index = weather_df_2017.index.tz_localize("UTC").tz_convert("America/Sao_Paulo")


    weather_df_2016 = pd.read_csv('../input/INMET_SE_RJ_A618_TERESOPOLIS_01-01-2016_A_31-12-2016.CSV', sep=';',parse_dates={'dt': ['DATA (YYYY-MM-DD)', 'HORA (UTC)']}, low_memory=False, decimal=',', na_values=['nan','?'], index_col='dt',skiprows=8, encoding='ISO-8859-1')
    weather_df_2016.columns = columns
    weather_df_2016.index = weather_df_2016.index.tz_localize("UTC").tz_convert("America/Sao_Paulo")

    weather_df = pd.concat([weather_df_2016,weather_df_2017,weather_df_2018, weather_df_2019, weather_df_2020, weather_df_2021, weather_df_2022, weather_df_2023])

    weather_df.to_csv('weather_2016_2023.csv')

if __name__ == "__main__":
    consolidate()