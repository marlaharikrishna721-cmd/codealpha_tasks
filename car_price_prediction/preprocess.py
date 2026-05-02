import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Drop unnecessary column
    df = df.drop(['Car_Name'], axis=1)

    # Convert Year to Car Age
    df['Car_Age'] = 2025 - df['Year']
    df.drop(['Year'], axis=1, inplace=True)

    # Convert categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df