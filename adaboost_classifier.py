import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
train_dataset_path = 'train.csv'
train_dataset = pd.read_csv(train_dataset_path)
test_dataset_path = 'test.csv'
test_dataset = pd.read_csv(test_dataset_path)

# Separate features and labels for both training and testing sets
x_train = train_dataset.drop('fake', axis=1)
y_train = train_dataset['fake']
x_test = test_dataset.drop('fake', axis=1)
y_test = test_dataset['fake']

# Apply PCA with 9 components
n_components = 9
pca = PCA(n_components=n_components)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Gradient Boosting Classifier without PCA
adaboost_classifier = AdaBoostClassifier(n_estimators=100, learning_rate=0.9, random_state=42)
adaboost_classifier.fit(x_train, y_train)
adaboost_classifier_prediction = adaboost_classifier.predict(x_test)

# Gradient Boosting Classifier with PCA
adaboost_classifier_pca = AdaBoostClassifier(n_estimators=100, learning_rate=0.9, random_state=42)
adaboost_classifier_pca.fit(x_train_pca, y_train)
adaboost_classifier_pca_prediction = adaboost_classifier_pca.predict(x_test_pca)

# Evaluate the performance without PCA
print("Gradient Boosting Classifier Results (Without PCA):")
print("Classification Report:\n", classification_report(y_test, adaboost_classifier_prediction))
print("Accuracy:", accuracy_score(y_test, adaboost_classifier_prediction))

# Evaluate the performance with PCA
print("\nGradient Boosting Classifier Results (With PCA):")
print("Classification Report:\n", classification_report(y_test, adaboost_classifier_pca_prediction))
print("Accuracy:", accuracy_score(y_test, adaboost_classifier_pca_prediction))

# Generate heatmaps
plt.figure(figsize=(15, 6))

# Heatmap without PCA
plt.subplot(1, 2, 1)
sns.heatmap(pd.crosstab(y_test, adaboost_classifier_prediction, rownames=['Actual'], colnames=['Predicted']), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Without PCA)')

# Heatmap with PCA
plt.subplot(1, 2, 2)
sns.heatmap(pd.crosstab(y_test, adaboost_classifier_pca_prediction, rownames=['Actual'], colnames=['Predicted']), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (With PCA)')

plt.show()
