from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def split_data(df):
    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model