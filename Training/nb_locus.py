import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

'''
(10/11 | Daksh)

This script implements the Naive Bayes (NB) algorithm for classification.

In this implementation:
- The input data includes the following features:
  1. COUNTRY: Country of the respondent
  2. GENDER: Gender of the respondent
  3. MARITAL: Marital status
  4. EMPLOYMENT: Employment status
  5. INCOME: Annual income
  6. RELG_NOW: Current religious belief
  7. SBS: Subjective belief system
  8. AFFILIATION: Political or religious affiliation
  9. AGE: Age of the respondent

- The target variable is 'LOCUS_Class', which is categorized into 'low', 'medium', and 'high' 
  based on the 'LOCUS_Sum' feature.

- The data is split into training and testing sets to evaluate the classifier's performance.

- The script includes an evaluation of the model using accuracy, confusion matrix, and 
  classification report metrics.

- A heatmap visualization of the confusion matrix is also provided for better interpretability 
  of the results.

'''

data = pd.read_csv('../CLEAN_DATA.csv')

def categorize_locus(value):
    if -24 <= value <= -9:
        return 'low'
    elif -8 <= value <= 7:
        return 'medium'
    elif 8 <= value <= 24:
        return 'high'

data['LOCUS_Class'] = data['LOCUS_Sum'].apply(categorize_locus)

features = ['COUNTRY', 'GENDER', 'MARITAL', 'EMPLOYMENT', 'INCOME',
            'RELG_NOW', 'SBS', 'AFFILIATION', 'AGE']
X = data[features]
y = data['LOCUS_Class']

label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X.loc[:, column] = le.fit_transform(X[column])
    label_encoders[column] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
class_report_nb = classification_report(y_test, y_pred_nb)

print(f'Naive Bayes Accuracy: {accuracy_nb:.2f}')
print('Naive Bayes Confusion Matrix:\n', conf_matrix_nb)
print('Naive Bayes Classification Report:\n', class_report_nb)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Naive Bayes Confusion Matrix')
plt.show()
