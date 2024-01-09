import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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

classifier1 = LogisticRegression(max_iter = 1000)
classifier2 = DecisionTreeClassifier()
classifier3 = RandomForestClassifier()

voting_classifier_hard = VotingClassifier(estimators=[
    ('logistic_regression', classifier1),
    ('decision_tree', classifier2),
    ('random_forest', classifier3)
    ], voting='soft')

voting_classifier_hard.fit(x_train,y_train)
voting_classifier_prediction = voting_classifier_hard.predict(x_test)

print("Prediction : ",voting_classifier_prediction)
print("Accuracy : ",accuracy_score(y_test,voting_classifier_prediction))
