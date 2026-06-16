# ✅ Correct imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# 📥 Load dataset (adjust path if needed)
# Assumes tourism.csv is inside a "data" folder at repo root
data = pd.read_csv("data/tourism.csv")

# 🧹 Drop identifier columns if present
data = data.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore")

# 🎯 Define target and features
X = data.drop("ProdTaken", axis=1)
y = data["ProdTaken"]

# 🔄 One-hot encode categorical features
X = pd.get_dummies(X)

# ✂️ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🤖 Train logistic regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 💾 Save model to repo root
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")
