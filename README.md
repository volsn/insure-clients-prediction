Setup info:

virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
python model.py

Input:
input.csv

Output:
output.json

in format
{
    key1: 1,
    key2: 0,
    key3: 0,
    key4: 1,
}

Dependecies:

model.pkl - Pickled trained machine learning model
sport_types.npy - Data needed for transform of new data
