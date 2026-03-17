from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import joblib

def train_model():
    df = pd.read_csv('data/processed_health_data.csv')

    # Features: What the model looks at
    features = ['TotalSteps', 'VeryActiveMinutes', 'SedentaryMinutes', 'Calories']
    X = df[features]
    y = df['RecoveryScore']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    print(f"Model Error: {round(error, 2)} points on the 0-100 scale.")

    joblib.dump(model, 'recovery_model.pkl')
    print("Model saved as recovery_model.pkl")

if __name__ == "__main__":
    train_model()