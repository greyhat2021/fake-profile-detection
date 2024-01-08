import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the datasets
train_dataset_path = 'train.csv'
test_dataset_path = 'test.csv'

# Load datasets
train_dataset = pd.read_csv(train_dataset_path)
test_dataset = pd.read_csv(test_dataset_path)

# Separate features and labels for both training and testing sets
x_train = train_dataset.drop('fake', axis=1)
y_train = train_dataset['fake']
x_test = test_dataset.drop('fake', axis=1)
y_test = test_dataset['fake']

# Decision Tree Classifier without PCA
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_prediction = dtc.predict(x_test)

# Decision Tree Classifier with PCA
n_comp = 9
pca = PCA(n_components=n_comp)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

dtc_pca = DecisionTreeClassifier()
dtc_pca.fit(x_train_pca, y_train)
dtc_pca_prediction = dtc_pca.predict(x_test_pca)

# Print prediction for both without and with PCA
print("Prediction without PCA: ", dtc_prediction)
print("Prediction with PCA: ", dtc_pca_prediction)

# Print accuracy and classification report for Decision Tree Classifier without PCA
print("\nAccuracy (DTC without PCA): ", accuracy_score(y_test, dtc_prediction))
print("Classification Report (DTC without PCA):\n", classification_report(y_test, dtc_prediction))

# Generate confusion matrix for Decision Tree Classifier without PCA
cm = confusion_matrix(y_test, dtc_prediction)

# Print accuracy and classification report for Decision Tree Classifier with PCA
print("\nAccuracy (DTC with PCA (",n_comp,")): ", accuracy_score(y_test, dtc_pca_prediction))
print("Classification Report (DTC with PCA):\n", classification_report(y_test, dtc_pca_prediction))

# Generate confusion matrix for Decision Tree Classifier with PCA
cm_pca = confusion_matrix(y_test, dtc_pca_prediction)

# Plot both heatmaps side by side using Seaborn
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot heatmap for Decision Tree Classifier without PCA
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['True Negative', 'True Positive'],
            yticklabels=['True Negative', 'True Positive'], ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix (DTC without PCA)')

# Plot heatmap for Decision Tree Classifier with PCA
sns.heatmap(cm_pca, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['True Negative', 'True Positive'],
            yticklabels=['True Negative', 'True Positive'], ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix (DTC with PCA)')

plt.show()
