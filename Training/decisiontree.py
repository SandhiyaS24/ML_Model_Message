import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('CLEAN_DATA.csv')

# Turning LOCUS_Sum values into classes
def categorize_locus(value):
    if -24 <= value <= -9:
        return 'low'
    elif -8 <= value <= 7:
        return 'medium'
    elif 8 <= value <= 24:
        return 'high'
data['LOCUS_Class'] = data['LOCUS_Sum'].apply(categorize_locus)

# Prepare features and target
    # 62% Accuracy
features = ['COUNTRY', 'GENDER', 'MARITAL', 'EMPLOYMENT', 'INCOME',
           'RELG_NOW', 'SBS', 'AFFILIATION', 'AGE']
''' 60% Accuracy
features = [ 'RELG_NOW', 'SPIRIT_NOW', 'SBS', 
           'RELBEH_Sum', 'GOD_pSum'
]
'''
X = data[features]
y = data['LOCUS_Class']

# Encode categorical variables
le = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = le.fit_transform(X[column])

# Training and testing classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# Confusion Matrix
class_labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=class_labels)
print("\nConfusion Matrix:")
print(cm)
class_labels = sorted(y.unique())
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion-matrix_decision-tree.jpeg', format='jpeg')
plt.show()

# Classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
feature_importance = dt.feature_importances_
root_node = features[np.argmax(feature_importance)]
print(f"\nRoot Node: {root_node}")
print("\nFeature Importance (Information Gain):")
for feature, importance in zip(features, feature_importance):
    print(f"{feature}: {importance:.4f}")

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=features, class_names=dt.classes_, filled=True, rounded=True)
plt.show()