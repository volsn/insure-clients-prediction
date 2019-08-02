# Predict

A part of module designed to predict labels for new data


## Setup info:

```bash
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
python predict.py
```

## Dependecies

```
model.pkl - Pickled trained machine learning model
sport_types.npy - Data needed for transform of new data
```

## Input
```
input.csv
when "csv" is in argv

otherwise

input.txt
In format:
161708|CALCULATE|ABS|1321.92|100000.0|ALWAYS|False|1554918514|2019-04-11|2020-04-10|||False|KHUDOZHESTVENNAYA-GIMNASTIKA|0|1|217.118.81.194|https://prosto.insure/sportivnaja-strakhovka/|"Mozilla/5.0 (Linux; Android 7.0; SLA-L22 Build/HUAWEISLA-L22) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.91 Mobile Safari/537.36"|180.0|False
| used as delimiter
```

## Output
```json
{
  "is_purchase": "1",
  "probabilities": [
    [
      "1255.824",
      "0.6"
    ],
    [
      "1076.712102",
      "0.6"
    ],
    [
      "876.9887365296374",
      "0.8"
    ],
    [
      "612.4339429727811",
      "1.0"
    ],
    [
      "552.7216335329349",
      "0.8"
    ]
  ]
}
```
