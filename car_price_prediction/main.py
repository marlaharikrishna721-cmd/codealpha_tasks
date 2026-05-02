# 🚗 Car Price Prediction (Single File - No src)

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 2: Load Dataset
df = pd.read_csv("car_data.csv")

print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())


# Step 3: Data Preprocessing

# Drop Car Name (not useful for prediction)
if 'Car_Name' in df.columns:
    df = df.drop(['Car_Name'], axis=1)

# Convert Year to Car Age
if 'Year' in df.columns:
    df['Car_Age'] = 2025 - df['Year']
    df.drop(['Year'], axis=1, inplace=True)

# Convert categorical columns into numbers
df = pd.get_dummies(df, drop_first=True)

print("\nProcessed Data:\n", df.head())


# Step 4: Visualization (Optional but useful)

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# Step 5: Split Data

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 6: Train Model

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")


# Step 7: Prediction

y_pred = model.predict(X_test)


# Step 8: Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# Step 9: Plot Actual vs Predicted

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()


# Step 10: Sample Prediction

sample = X.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample)

print("\nSample Predicted Price:", prediction[0])