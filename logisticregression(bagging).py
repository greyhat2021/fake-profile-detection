import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

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

base_model = LogisticRegression(max_iter=450)
bagging_classifier_logisticregression = BaggingClassifier(base_model, n_estimators=10, random_state=42)

# Without PCA
bagging_classifier_logisticregression.fit(x_train, y_train)
bagging_classifier_logisticregression_prediction = bagging_classifier_logisticregression.predict(x_test)

# With PCA
n_comp = 9
pca = PCA(n_components=n_comp)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

bagging_classifier_logisticregression.fit(x_train_pca, y_train)
bagging_classifier_logisticregression_pca_prediction = bagging_classifier_logisticregression.predict(x_test_pca)

# Print and plot results
print("Prediction without PCA:", bagging_classifier_logisticregression_prediction)
print("Classification report without PCA:\n", classification_report(y_test, bagging_classifier_logisticregression_prediction))
print(f'Accuracy without PCA: {accuracy_score(y_test, bagging_classifier_logisticregression_prediction)}')

print("Prediction with PCA:", bagging_classifier_logisticregression_pca_prediction)
print("Classification report with PCA:\n", classification_report(y_test, bagging_classifier_logisticregression_pca_prediction))
print(f'Accuracy with PCA: {accuracy_score(y_test, bagging_classifier_logisticregression_pca_prediction)}')

# Create heatmaps
plt.figure(figsize=(15, 6))

# Without PCA
plt.subplot(1, 2, 1)
sns.heatmap(pd.crosstab(y_test, bagging_classifier_logisticregression_prediction, rownames=['Actual'], colnames=['Predicted']), annot=True, fmt='d', cmap='Blues')

plt.title('Confusion Matrix (Without PCA)')

# With PCA
plt.subplot(1, 2, 2)
sns.heatmap(pd.crosstab(y_test, bagging_classifier_logisticregression_pca_prediction, rownames=['Actual'], colnames=['Predicted']), annot=True, fmt='d', cmap='Blues')

plt.title('Confusion Matrix (With PCA)')

plt.show()
