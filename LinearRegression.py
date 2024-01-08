import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.decomposition import PCA

# Assuming 'fake' is your target variable
train_dataset_path = 'train.csv'
train_dataset = pd.read_csv(train_dataset_path)
test_dataset_path = 'test.csv'
test_dataset = pd.read_csv(test_dataset_path)

# Separate features and labels for both training and testing sets
x_train = train_dataset.drop('fake', axis=1)
y_train = train_dataset['fake']
x_test = test_dataset.drop('fake', axis=1)
y_test = test_dataset['fake']

# Linear Regression without PCA
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

# Predict continuous values
linear_predictions = linear_model.predict(x_test)

# Apply threshold of 0.75 and handle negative values
threshold = 0.75
binary_predictions = (linear_predictions >= threshold).astype(int)
binary_predictions[linear_predictions < 0] = 0  # Set negative predictions to 0
print("Linear Predictions without PCA: \n", linear_predictions)
print("Binary Predictions without PCA (Threshold 0.75): \n", binary_predictions)

# Evaluate using mean squared error (MSE)
mse = mean_squared_error(y_test, binary_predictions)
print("Mean Squared Error without PCA: ", mse)
print("Accuracy without PCA: ", accuracy_score(y_test, binary_predictions))

# Generate confusion matrix without PCA
cm = confusion_matrix(y_test, binary_predictions)

# Classification report without PCA
classification_rep = classification_report(y_test, binary_predictions)
print("Classification Report without PCA:\n", classification_rep)

# Linear Regression with PCA
n_comp = 9
pca = PCA(n_components=n_comp)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

linear_model_pca = LinearRegression()
linear_model_pca.fit(x_train_pca, y_train)

# Predict continuous values with PCA
linear_predictions_pca = linear_model_pca.predict(x_test_pca)

# Apply threshold of 0.75 and handle negative values with PCA
binary_predictions_pca = (linear_predictions_pca >= threshold).astype(int)
binary_predictions_pca[linear_predictions_pca < 0] = 0  # Set negative predictions to 0
print("\nLinear Predictions with PCA: \n", linear_predictions_pca)
print("Binary Predictions with PCA (Threshold 0.75): \n", binary_predictions_pca)

# Evaluate using mean squared error (MSE) with PCA
mse_pca = mean_squared_error(y_test, binary_predictions_pca)
print("Mean Squared Error with PCA (", n_comp, "): ", mse_pca)
print("Accuracy with PCA (", n_comp, "): ", accuracy_score(y_test, binary_predictions_pca))

# Generate confusion matrix with PCA
cm_pca = confusion_matrix(y_test, binary_predictions_pca)

# Classification report with PCA
classification_rep_pca = classification_report(y_test, binary_predictions_pca)
print("\nClassification Report with PCA:\n", classification_rep_pca)

# Plot both heatmaps side by side using Seaborn
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot heatmap without PCA
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['True Negative', 'True Positive'],
            yticklabels=['True Negative', 'True Positive'], ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix without PCA')

# Plot heatmap with PCA
sns.heatmap(cm_pca, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['True Negative', 'True Positive'],
            yticklabels=['True Negative', 'True Positive'], ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix with PCA')

plt.show()
