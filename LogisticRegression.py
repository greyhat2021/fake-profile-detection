import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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

# Logistic Regression without PCA
logreg = LogisticRegression(max_iter=250)
logreg.fit(x_train, y_train)
logreg_prediction = logreg.predict(x_test)

# Print prediction without PCA
print("Prediction without PCA: ", logreg_prediction)

# Print accuracy and classification report for Logistic Regression without PCA
print("\nAccuracy logreg without PCA: ", accuracy_score(y_test, logreg_prediction))
print("Classification Report logreg without PCA:\n", classification_report(y_test, logreg_prediction))

# Generate confusion matrix for Logistic Regression without PCA
cm = confusion_matrix(y_test, logreg_prediction)

# Logistic Regression with PCA
n_comp = 9
pca = PCA(n_components=n_comp)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

logreg_pca = LogisticRegression(max_iter=250)
logreg_pca.fit(x_train_pca, y_train)
logreg_pca_prediction = logreg_pca.predict(x_test_pca)

# Print prediction with PCA
print("\nPrediction with PCA: ", logreg_pca_prediction)

# Print accuracy and classification report for Logistic Regression with PCA
print("\nAccuracy logreg with PCA (", n_comp, "): ", accuracy_score(y_test, logreg_pca_prediction))
print("Classification Report logreg with PCA:\n", classification_report(y_test, logreg_pca_prediction))

# Generate confusion matrix for Logistic Regression with PCA
cm_pca = confusion_matrix(y_test, logreg_pca_prediction)

# Plot both heatmaps side by side using Seaborn
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot heatmap for Logistic Regression without PCA
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['True Negative', 'True Positive'],
            yticklabels=['True Negative', 'True Positive'], ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix (LogReg without PCA)')

# Plot heatmap for Logistic Regression with PCA
sns.heatmap(cm_pca, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['True Negative', 'True Positive'],
            yticklabels=['True Negative', 'True Positive'], ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix (LogReg with PCA)')

plt.show()
