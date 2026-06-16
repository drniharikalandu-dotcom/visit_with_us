import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from huggingface_hub import hf_hub_download

# Load dataset from Hugging Face Hub
repo_id = "DrNiha555/visit_with_us"  # dataset repo
filename = "tourism.csv"
dataset_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
data = pd.read_csv(dataset_path)

# Drop identifier columns if present
data = data.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore")

# Define target and features
X = data.drop("ProdTaken", axis=1)
y = data["ProdTaken"]

# One-hot encode categorical features
X = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save artifacts directly to repo root
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model trained and saved as model.pkl and scaler.pkl")

