import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('Crop_2.csv')

# Features and target variable (removed temperature, humidity, and rainfall)
X = data[['N', 'P', 'K', 'ph']]
y = data['label']  # Ensure this column contains the crop types

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model's performance
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Function to get user input and predict crop987
def predict_crop():
    print("Please enter the following soil parameters:")
    N = float(input("Nitrogen (N) in ppm: "))
    P = float(input("Phosphorus (P) in ppm: "))
    K = float(input("Potassium (K) in ppm: "))
    ph = float(input("pH level of soil: "))

    # Create a DataFrame for the new input
    new_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'ph': [ph]
    })

    # Predict the crop type
    predicted_crop = model.predict(new_data)
    print(f'Suggested Crop: {predicted_crop[0]}')

# Call the function to get user input and make a prediction
predict_crop()
