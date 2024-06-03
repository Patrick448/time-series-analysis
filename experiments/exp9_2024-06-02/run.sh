python ../../main.py -cf config1.json -oh
python ../../main.py -cf config1.json
python ../../main.py -cf config1.json
python ../../main.py -cf config1.json
python ../../main.py -cf config1.json

python ../../main.py -cf config2.json
python ../../main.py -cf config2.json
python ../../main.py -cf config2.json
python ../../main.py -cf config2.json
python ../../main.py -cf config2.json

python ../../main.py -cf config3.json
python ../../main.py -cf config3.json
python ../../main.py -cf config3.json
python ../../main.py -cf config3.json
python ../../main.py -cf config3.json

python ../../scripts/create_error_charts.py -p result.csv -o analysis -cd
