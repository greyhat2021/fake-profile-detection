import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# RandomForestClassifier without PCA
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_prediction = rfc.predict(x_test)

# Print prediction without PCA
print("Prediction without PCA: ", rfc_prediction)

# Print accuracy and classification report for RandomForestClassifier without PCA
print("\nAccuracy RFC without PCA: ", accuracy_score(y_test, rfc_prediction))
print("Classification Report RFC without PCA:\n", classification_report(y_test, rfc_prediction))

# Generate confusion matrix for RandomForestClassifier without PCA
cm = confusion_matrix(y_test, rfc_prediction)

# RandomForestClassifier with PCA
n_comp = 9
pca = PCA(n_components=n_comp)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

rfc_pca = RandomForestClassifier()
rfc_pca.fit(x_train_pca, y_train)
rfc_pca_prediction = rfc_pca.predict(x_test_pca)

# Print prediction with PCA
print("\nPrediction with PCA: ", rfc_pca_prediction)

# Print accuracy and classification report for RandomForestClassifier with PCA
print("\nAccuracy RFC with PCA (", n_comp, "): ", accuracy_score(y_test, rfc_pca_prediction))
print("Classification Report RFC with PCA:\n", classification_report(y_test, rfc_pca_prediction))

# Generate confusion matrix for RandomForestClassifier with PCA
cm_pca = confusion_matrix(y_test, rfc_pca_prediction)

# Plot both heatmaps side by side using Seaborn
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot heatmap for RandomForestClassifier without PCA
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['True Negative', 'True Positive'],
            yticklabels=['True Negative', 'True Positive'], ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix (RFC without PCA)')

# Plot heatmap for RandomForestClassifier with PCA
sns.heatmap(cm_pca, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['True Negative', 'True Positive'],
            yticklabels=['True Negative', 'True Positive'], ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix (RFC with PCA)')

plt.show()
