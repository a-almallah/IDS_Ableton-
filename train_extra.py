import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("Loading dataset for extra models...")
df = pd.read_csv('Dataset/CSE-CIC-IDS2018/Bot.csv', nrows=100000)
df.dropna(inplace=True)
df = df.replace([float('inf'), float('-inf')], 0)
df.columns = df.columns.str.strip()

X = df.drop('Label', axis=1)
y = df['Label']

# Drop time features to match scaler
time_features = [col for col in X.columns if 'Duration' in col or 'IAT' in col]
X = X.drop(columns=time_features, errors='ignore')

# Use existing scaler to ensure compatibility
scaler = joblib.load('scaler.pkl')
X_scaled = scaler.transform(X)

print("Training Decision Tree...")
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_scaled, y)
joblib.dump(dt, 'dt_model.pkl')

print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, y)
joblib.dump(lr, 'lr_model.pkl')

print("Done training extra models.")
