from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Import csv data
data = pd.read_csv('training_set.csv')

# Convert 'SEX' column into numerical datatype
data['SEX'] = data['SEX'].astype('category').cat.codes

# Define the features and labels
features = ['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE', 'THROMBOCYTE', 'MCH', 'MCHC', 'MCV', 'AGE', 'SEX']
label = 'SOURCE'

# Create a 5-fold cross-validation object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the percentages of data for training, validation, and testing
train_size = 0.7
val_size = 0.1
test_size = 0.2

# Loop over each fold
for i, (train_index, test_index) in enumerate(kf.split(data)):
    print(f"Fold {i+1}")

    # Split the data into features and labels
    X = data[features]
    y = data[label]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

    # Split the training set into training, validation, and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_size, test_size=val_size+test_size, random_state=42)

    # # Print the sizes of the train-val-test sets for this fold
    # print("Training set size:", len(X_train))
    # print("Validation set size:", len(X_val))
    # print("Testing set size:", len(X_test))

    # Create a Naive Bayes classifier
    clf = GaussianNB()

    # Train the classifier on the training set
    clf.fit(X_train, y_train)

    # Use the trained classifier to make predictions on the validation set
    y_pred = clf.predict(X_val)

    # Print the classification report for the validation set
    print("Validation Set:")
    print(classification_report(y_val, y_pred))

    # Use the trained classifier to make predictions on the test set
    y_pred = clf.predict(X_test)

    # Print the classification report for the test set
    print("Test Set:")
    print(classification_report(y_test, y_pred))
