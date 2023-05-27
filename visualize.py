import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "./data/predictions_2.csv"

def get_mistake_data():
    df = pd.read_csv(DATA_PATH)
    
    # in the cols where status is "wrong" get the value of label and prediction
    df = df[df["status"] == "❌"]
    df = df[["label", "prediction"]]
    
    return df
    
mistakes_df = get_mistake_data()

def absolute_value(val):
    a  = np.round(val/100.*len(mistakes_df), 0)
    return a

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.pie(mistakes_df["label"].value_counts(), labels=mistakes_df["label"].value_counts().keys(), autopct=absolute_value)
ax.set_title(f"Pie Chart of Mistakes | Count: {len(mistakes_df)}")
plt.show()