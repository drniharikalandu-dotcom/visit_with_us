import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load training data
train_df = pd.read_csv("data/train.csv")

# Show columns for debugging
print("Columns in train.csv:", train_df.columns)

# Replace 'Purchased' with your actual target column name
target_col = "Purchased"   # <-- change this to match your dataset

X = train_df.drop(target_col, axis=1)
y = train_df[target_col]

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model training complete. Saved as model.pkl")
