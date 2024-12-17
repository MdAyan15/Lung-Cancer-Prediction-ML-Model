import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
data = pd.read_csv('survey lung cancer.csv')

# Clean column names by stripping any leading/trailing spaces
data.columns = data.columns.str.strip()

# Initialize LabelEncoder
le = LabelEncoder()

# List of categorical columns (added cleaned names)
categorical_columns = ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
                       'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 
                       'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 
                       'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 
                       'CHEST PAIN']

# Encode categorical columns
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Features and target variable
X = data.drop('LUNG_CANCER', axis=1)  # Drop the target column
y = data['LUNG_CANCER']  # Target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('Lung_Cancer.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model has been trained and saved as 'Lung_Cancer.pkl'")
