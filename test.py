import pandas as pd
import joblib
import argparse
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="Test IDS model on unseen data.")
    parser.add_argument("--attack", type=str, required=True, help="Keyword of the attack type that was trained (e.g., DoS, DDoS, Bot)")
    parser.add_argument("--model-name", type=str, default="model.pkl", help="Model filename to load")
    args = parser.parse_args()

    test_file = f'test_data_{args.attack}.csv'
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test data '{test_file}' not found. Please run train.py --attack {args.attack} first.")
        
    print(f"Loading test dataset '{test_file}'...")
    df = pd.read_csv(test_file)
    
    if 'Label' not in df.columns:
        raise ValueError("Test dataset must contain a 'Label' column.")
        
    X_test = df.drop('Label', axis=1)
    y_test = df['Label']

    print(f"Loading models ({args.model_name}, scaler.pkl)...")
    clf = joblib.load(args.model_name)
    scaler = joblib.load('scaler.pkl')
    model_features = joblib.load('model_features.pkl')

    # Ensure columns match exactly
    X_test = X_test[model_features]

    print("Scaling test data...")
    X_test_scaled = scaler.transform(X_test)
    
    print("\nRunning predictions...")
    preds = clf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, preds)
    print(f"\n==============================")
    print(f" Test Accuracy: {acc:.4f}")
    print(f"==============================\n")
    
    print("Classification Report:")
    print(classification_report(y_test, preds))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    main()
