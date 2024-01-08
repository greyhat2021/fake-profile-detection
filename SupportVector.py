import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
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

# Support Vector Machine without PCA
svm = SVC()
svm.fit(x_train, y_train)
svm_prediction = svm.predict(x_test)

# Print prediction without PCA
print("Prediction without PCA: ", svm_prediction)

# Print accuracy and classification report for SVM without PCA
print("\nAccuracy SVM without PCA: ", accuracy_score(y_test, svm_prediction))
print("Classification Report SVM without PCA:\n", classification_report(y_test, svm_prediction))

# Generate confusion matrix for SVM without PCA
cm = confusion_matrix(y_test, svm_prediction)

# Support Vector Machine with PCA
n_comp = 9
pca = PCA(n_components=n_comp)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

svm_pca = SVC()
svm_pca.fit(x_train_pca, y_train)
svm_pca_prediction = svm_pca.predict(x_test_pca)

# Print prediction with PCA
print("\nPrediction with PCA: ", svm_pca_prediction)

# Print accuracy and classification report for SVM with PCA
print("\nAccuracy SVM with PCA (", n_comp, "): ", accuracy_score(y_test, svm_pca_prediction))
print("Classification Report SVM with PCA:\n", classification_report(y_test, svm_pca_prediction))

# Generate confusion matrix for SVM with PCA
cm_pca = confusion_matrix(y_test, svm_pca_prediction)

# Plot both heatmaps side by side using Seaborn
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot heatmap for SVM without PCA
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['True Negative', 'True Positive'],
            yticklabels=['True Negative', 'True Positive'], ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix (SVM without PCA)')

# Plot heatmap for SVM with PCA
sns.heatmap(cm_pca, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['True Negative', 'True Positive'],
            yticklabels=['True Negative', 'True Positive'], ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix (SVM with PCA)')

plt.show()
