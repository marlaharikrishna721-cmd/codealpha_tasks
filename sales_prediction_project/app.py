import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_preprocessing import load_data, clean_data, encode_data, split_features
from train_model import train
from evaluate_model import evaluate

st.title("📊 Sales Prediction Dashboard")

# Load data
df = load_data("data_csv.csv")

st.subheader("📄 Dataset Preview")
st.write(df.head())

# Graphs
st.subheader("📊 Data Visualization")

fig1, ax1 = plt.subplots()
sns.scatterplot(x=df["TV"], y=df["Sales"], ax=ax1)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.scatterplot(x=df["Radio"], y=df["Sales"], ax=ax2)
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# Clean + prepare
df = clean_data(df)
df = encode_data(df)

X, y = split_features(df)

# Train model
model, X_test, y_test = train(X, y)

# Evaluate
mae, r2 = evaluate(model, X_test, y_test)

st.subheader("📊 Model Performance")
st.write(f"MAE: {mae}")
st.write(f"R2 Score: {r2}")

# Feature importance
st.subheader("📈 Feature Importance")

if hasattr(model, "coef_"):
    importance = model.coef_
else:
    importance = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Impact": importance
})

st.write(feat_df)

# Prediction input
st.subheader("🔮 Make Prediction")

tv = st.slider("TV Advertising", 0, 300, 100)
radio = st.slider("Radio Advertising", 0, 50, 20)
news = st.slider("Newspaper Advertising", 0, 100, 30)

input_df = pd.DataFrame({
    "TV": [tv],
    "Radio": [radio],
    "Newspaper": [news]
})

prediction = model.predict(input_df)

st.write("### Predicted Sales:", prediction[0])