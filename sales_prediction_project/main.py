from data_preprocessing import load_data, clean_data, encode_data, split_features
from train_model import train
from evaluate_model import evaluate
from predict import make_prediction

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # ✅ Load dataset (make sure filename matches your file)
    df = load_data("data_csv.csv")

    print("\n📄 Data Preview:")
    print(df.head())

    # ✅ Graphs (EDA)
    print("\n📊 Generating Graphs...")

    sns.scatterplot(x=df["TV"], y=df["Sales"])
    plt.title("TV vs Sales")
    plt.show()

    sns.scatterplot(x=df["Radio"], y=df["Sales"])
    plt.title("Radio vs Sales")
    plt.show()

    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # ✅ Clean data
    df = clean_data(df)

    # ✅ Encode (safe step)
    df = encode_data(df)

    # ✅ Split features
    X, y = split_features(df)

    # ✅ Train model
    model, X_test, y_test = train(X, y)

    # ✅ Evaluate model
    mae, r2 = evaluate(model, X_test, y_test)

    print("\n📊 Model Performance:")
    print("MAE:", mae)
    print("R2 Score:", r2)

    # ✅ Feature Importance (works for BOTH models)
    print("\n📈 Feature Impact:")

    if hasattr(model, "coef_"):
        coeff = pd.DataFrame(model.coef_, X.columns, columns=["Impact"])
    else:
        coeff = pd.DataFrame(model.feature_importances_, X.columns, columns=["Impact"])

    print(coeff)

    # ✅ Prediction
    sample = X.iloc[0:1]
    prediction = make_prediction(model, sample)

    print("\n🔮 Sample Prediction:", prediction)


if __name__ == "__main__":
    main()