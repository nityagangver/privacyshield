import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def main():
    print("Loading dataset inside setup_scaler...")
    # Load dataset from parent directory
    df = pd.read_csv('../WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Preprocess exactly the same way as main.py
    # Fix TotalCharges — has blank strings for new customers
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Drop CustomerID — not a feature
    df = df.drop('customerID', axis=1, errors='ignore')

    # Encode target
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # Encode all categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Features and target
    X = df.drop('Churn', axis=1)
    
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved successfully to models/scaler.pkl")

if __name__ == '__main__':
    main()
