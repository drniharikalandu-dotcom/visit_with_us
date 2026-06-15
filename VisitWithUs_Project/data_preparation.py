import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset from data folder
df = pd.read_csv("data/tourism.csv")

# Basic cleaning
df = df.drop_duplicates()

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save splits back into data folder
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Data preparation complete. Train and test sets saved.")
git add data/tourism.csv
git commit -m "Add tourism.csv dataset"
git push origin main

