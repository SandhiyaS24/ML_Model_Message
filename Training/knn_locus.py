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

# Turn LOCUS_Sum values into classes
def categorize_locus(value):
    if -24 <= value <= -9:
        return 'low'
    elif -8 <= value <= 7:
        return 'medium'
    elif 8 <= value <= 24:
        return 'high'

data['LOCUS_Class'] = data['LOCUS_Sum'].apply(categorize_locus)

# Prepare features and target
features = ['COUNTRY', 'GENDER', 'MARITAL', 'EMPLOYMENT', 'INCOME',
            'RELG_NOW', 'SBS', 'AFFILIATION', 'AGE']
X = data[features]
y = data['LOCUS_Class']

# Encode categorical variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X.loc[:, column] = le.fit_transform(X[column])  # Use .loc to avoid the warning
    label_encoders[column] = le  # Store the encoder for possible inverse transformation later

# Training and testing classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
