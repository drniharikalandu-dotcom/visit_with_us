import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load training data
train_df = pd.read_csv("data/train.csv")

print("Columns in train.csv:", train_df.columns)

# Target column
target_col = "ProdTaken"

# Drop unwanted columns
X = train_df.drop([target_col, "Unnamed: 0", "CustomerID"], axis=1)
y = train_df[target_col]

# Convert categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X)

# Train logistic regression model
model = LogisticRegression(max_iter=500)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model training complete. Saved as model.pkl")
