import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/predictions_4.csv"
sns.set_theme(style="white")

def mistake_data_barchart():
    df = pd.read_csv(DATA_PATH)

    # in the cols where status is "wrong" get the value of label and prediction
    df = df[df["status"] == "❌"]
    df = df[["prediction"]]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.bar(df["prediction"].value_counts().keys(), df["prediction"].value_counts())
    plt.show()

def confidence_heatmap_chart():
    df = pd.read_csv(DATA_PATH)

    # in the cols where status is "wrong" get the value of label and prediction
    df = df[df["status"] == "❌"]
    df = df[["prediction", "neo_", "trinity_", "morpheus_"]]
    
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    
    ax[0][0].bar(df["prediction"].value_counts().keys(), df["prediction"].value_counts())
    ax[0][0].set_title("Prediction")
    ax[0][1].bar(df["neo_"].value_counts().keys(), df["neo_"].value_counts())
    ax[0][1].set_title("Neo")
    ax[1][0].bar(df["trinity_"].value_counts().keys(), df["trinity_"].value_counts())
    ax[1][0].set_title("Trinity")
    ax[1][1].bar(df["morpheus_"].value_counts().keys(), df["morpheus_"].value_counts())
    ax[1][1].set_title("Morpheus")
    plt.show()
    
    
    
    
confidence_heatmap_chart()


