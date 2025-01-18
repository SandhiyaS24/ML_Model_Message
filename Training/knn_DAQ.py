import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('CLEAN_DATA.csv')

# Check for NaN values in 'DAQ' and drop if any
data.dropna(subset=['DAQ'], inplace=True)

# Turn DAQ values into classes
""" Sandhiya -- min - 0.000000,25% - 2.066667, 50% - 3.866667, 75% - 5.266667, max - 8.000000 Since the DAQ values lie around 0 to 8. Encoding the values based on the percentiles."""
def categorize_locus(value):
    if 0 <= value <= 3:
        return 'low'
    elif 3 < value <= 5:
        return 'medium'
    elif 5 < value <= 8:
        return 'high'

data['DAQ_Class'] = data['DAQ'].apply(categorize_locus)

# Prepare features and target
features = ['COUNTRY', 'GENDER', 'MARITAL', 'EMPLOYMENT', 'INCOME',
            'RELG_NOW', 'SBS', 'AFFILIATION', 'AGE']
X = data[features]
y = data['DAQ_Class']

# Encode categorical variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X.loc[:, column] = le.fit_transform(X[column])  # Use .loc to avoid the warning
    label_encoders[column] = le  # Store the encoder for possible inverse transformation later

# Training and testing classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and fit the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # Changed to KNeighborsClassifier
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)

# Optional: Visualization of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
