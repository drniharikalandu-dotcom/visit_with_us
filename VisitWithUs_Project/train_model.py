import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load training data
train_df = pd.read_csv("data/train.csv")

# Example: assume target column is 'label'
X = train_df.drop("label", axis=1)
y = train_df["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model training complete. Saved as model.pkl")
