import pickle
import pandas as pd
import numpy as np

with open("xgb_credit_fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

AMOUNT_MEAN, AMOUNT_STD = 88.35, 250.12
TIME_MEAN,   TIME_STD   = 94813.86, 47488.15

test_cases = [
    {
        "label": "Fraud case 1 (expect >0.8)",
        "V1": -1.3598071, "V2": -0.0727812, "V3": 2.5363467, "V4": 1.3781552,
        "V5": -0.3383208, "V6": 0.4623878,  "V7": 0.2395986, "V8": 0.0986979,
        "V9": 0.3637870,  "V10": 0.0907942, "V11": -0.5515995,"V12": -0.6178009,
        "V13": -0.9913898,"V14": -0.3111694,"V15": 1.4681770, "V16": -0.4704005,
        "V17": 0.2079708, "V18": 0.0257906, "V19": 0.4039930, "V20": 0.2514121,
        "V21": -0.0183068,"V22": 0.2778376, "V23": -0.1104740,"V24": 0.0669281,
        "V25": 0.1285394, "V26": -0.1891484,"V27": 0.1335584, "V28": -0.0210530,
        "Amount": 149.62, "Time": 0,
    },
    {
        "label": "Legit case 1 (expect <0.05)",
        "V1": -0.9662717, "V2": -0.1852960, "V3": 1.7929490,  "V4": -0.8632913,
        "V5": -0.0103089, "V6": 1.2473040,  "V7": 0.2376090,  "V8": 0.3774910,
        "V9": -1.3870070, "V10": -0.0549518,"V11": -0.2264254, "V12": 0.1782820,
        "V13": 0.5077580, "V14": -0.2879080,"V15": -0.6314170, "V16": -1.0596000,
        "V17": -0.6842600,"V18": 1.9651690, "V19": -1.2324050, "V20": -0.2080830,
        "V21": -0.1083000,"V22": 0.0052736, "V23": -0.1902490, "V24": 0.3130530,
        "V25": 0.0274618, "V26": -0.1104320,"V27": 0.0669888,  "V28": 0.0285733,
        "Amount": 2.69, "Time": 406,
    },
]

V_FEATURES   = [f"V{i}" for i in range(1, 29)]
ALL_FEATURES = V_FEATURES + ["Amount_scaled", "Time_scaled"]

for case in test_cases:
    label  = case.pop("label")
    amount = case.pop("Amount")
    time   = case.pop("Time")

    case["Amount_scaled"] = (amount - AMOUNT_MEAN) / AMOUNT_STD
    case["Time_scaled"]   = (time   - TIME_MEAN)   / TIME_STD

    df   = pd.DataFrame([case])[ALL_FEATURES]
    prob = model.predict_proba(df)[0][1]
    flag = "FRAUD" if prob >= 0.5 else "LEGIT"

    print(f"{label}")
    print(f"  → Fraud probability: {prob:.4f}  |  Prediction: {flag}")
    print()