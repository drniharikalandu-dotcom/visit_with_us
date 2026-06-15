import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load training data
train_df = pd.read_csv("data/train.csv")

# Show columns for debugging
print("Columns in train.csv:", train_df.columns)

# Target column is 'ProdTaken'
target_col = "ProdTaken"

# Features (drop target + any unwanted index column)
X = train_df.drop([target_col, "Unnamed: 0", "CustomerID"], axis=1)
y = train_df[target_col]

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model training complete. Saved as model.pkl")

