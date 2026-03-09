import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
data = pd.read_csv("data/housing.csv")

print(data.head())

# Step 8: Features and target
X = data[['RM','LSTAT','PTRATIO']]
y = data['MEDV']

# Step 9: Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 10: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 11: Predictions
predictions = model.predict(X_test)

# Step 12: Evaluate model
error = mean_squared_error(y_test, predictions)
print("Model Error:", error)

# Step 13: Save model
pickle.dump(model, open("models/house_price_model.pkl", "wb"))

print("Model saved successfully!")