#python ../../main.py -if "../../processed_data/price_weather_weekly_df-2016-2023.csv" -m "V0" -is 1 -os 8 -rf experiment1.csv -c "Alface Crespa - Roça;TEMPERATURA DO PONTO DE ORVALHO (°C)" -save_path "../../saved_models" -oh
python ../../main.py -cf config.json -oh -is 1
python ../../main.py -cf config.json -is 2
python ../../main.py -cf config.json -is 3
python ../../main.py -cf config.json -is 4
python ../../main.py -cf config.json -is 5
python ../../main.py -cf config.json -is 6
python ../../main.py -cf config.json -is 7
python ../../main.py -cf config.json -is 8
