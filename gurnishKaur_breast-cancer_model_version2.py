import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import urllib.request

# URL of the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

# Download and load the dataset
raw_data = urllib.request.urlopen(url)

# Read the dataset into a DataFrame
df = pd.read_csv(raw_data, header=None)

# Split features and target variable
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

# Encode the target variable
y = [1 if val == 'M' else 0 for val in y]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)

# Train the model on the balanced dataset
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print the classification report
report = classification_report(y_test, y_pred)
print(report)