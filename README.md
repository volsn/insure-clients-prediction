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
input.csv
Being set through cmd:
``` cmd
python predict.py "4601" "SENT" "RES" "417.6" "300000.0" "ALWAYS" "False" "1559590087" "2019-06-09" "2019-06-09" "26.0" "nan" "False" "SHESTOVAYA-AKROBATIKA--POLDENS-POLE-DANCE-PILON-" "1" "0" "213.87.128.216" "https://prosto.insure/sportivnaja-strakhovka/sport/shestovaya-akrobatika--poldens-pole-dance-pilon-" "Mozilla/5.0 (Linux; Android 8.0.0; VTR-L29) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Mobile Safari/537.36" "180.0" "False" 
{"is_purchase": "0"}
```
Columns order
```
"key" "status" "company" "premium" "sum" "action_type" "is_partner" "created_at" "init_from" "init_till" "year" "place" "is_foreigner" "sports" "adult" "child" "ip" "referer" "user_agent" "timezone" "is_adblock_enabled"
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

# Train

A part of module designed for particial training of a model

```
Input in csv format must contain labels (y) in 'is_purchase' field and be name 'input.csv'
Particial training is done by built in function of SDGClassifier,
it can be replaced with any other sklearn model that has particial_fir method
```
