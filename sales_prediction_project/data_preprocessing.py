import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna()
    
    # remove unwanted column
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
        
    return df

def encode_data(df):
    return pd.get_dummies(df, drop_first=True)

def split_features(df):
    X = df.drop("Sales", axis=1)
    y = df["Sales"]
    return X, y