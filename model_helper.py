import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def get_control_score(x_train, y_train):
    mode = y_train.value_counts().idxmax()

    control_model = DummyClassifier(strategy = 'constant', constant = mode)
    control_model.fit(x_train, y_train)

    return control_model.score(x_train, y_train)


def compare_models(x_train, y_train, x_validate, y_validate):
    """
    It takes in the training and validation data, and then runs a random forest classifier with
    different max_depth and min_samples_leaf values. It then returns a dataframe with the results of
    each model.

    :param x_train: The training data
    :param y_train: The target variable for the training data
    :param x_validate: the validation data
    :param y_validate: the actual values of the target variable
    :return: A dataframe of all the model values
    """
    models_cont = []

    for num in range(2, 11):
        for val in range(1, 20):
            classifier = RandomForestClassifier(random_state = 125, max_depth = num, min_samples_leaf = val)
            classifier.fit(x_train, y_train)
            train_score = classifier.score(x_train, y_train)
            predictions = classifier.predict(x_validate)

            tp = confusion_matrix(y_validate, predictions)[1][1]
            fp = confusion_matrix(y_validate, predictions)[0][1]
            tn = confusion_matrix(y_validate, predictions)[0][0]
            fn = confusion_matrix(y_validate, predictions)[1][0]
            validate_score = classifier.score(x_validate, y_validate)

            output = {
                'max_depth':num,
                'min_samples_leaf': val,
                'True Positves': tp,
                'False Positives': fp,
                'True Negatives': tn,
                'False Negatvies': fn,
                'Precision': tp / (tp + fp),
                'Recall': tp / (tp + fn),
                'Training Acc Score': train_score,
                'Validate Acc Score': validate_score,
                'Acc Score Difference': train_score - validate_score
            }
            models_cont.append(output)
    return pd.DataFrame(models_cont)


def test_classifier(x_train, y_train, X_validate, y_validate, x_test, y_test):
    """
    The function takes in the training, validation, and test data and fits a random forest classifier to
    the training data. It then scores the model on the training, validation, and test data. It then
    predicts the test data and creates a confusion matrix. It then creates a dictionary of the
    evaluation parameters and returns the classifier and the dataframe of the evaluation parameters

    :param x_train: the training data
    :param y_train: the target variable for the training set
    :param X_validate: The validation set
    :param y_validate: the target variable for the validation set
    :param x_test: the test data
    :param y_test: the actual values of the target variable
    :return: The classifier and the test_df
    """

    classifier = RandomForestClassifier(random_state = 123, max_depth = 10, min_samples_leaf = 5)
    classifier.fit(x_train, y_train)
    train_score = classifier.score(x_train, y_train)
    validate_score = classifier.score(X_validate, y_validate)
    test_score = classifier.score(x_test, y_test)
    clf_preds = classifier.predict(x_test)
    tn = confusion_matrix(y_test, clf_preds)[0][0]
    fn = confusion_matrix(y_test, clf_preds)[1][0]
    tp = confusion_matrix(y_test, clf_preds)[1][1]
    fp = confusion_matrix(y_test, clf_preds)[0][1]

    eval_params = {
        'max_depth':10,
        'min_samples_leaf': 5,
        'True Positves': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatvies': fn,
        'Precision': tp / (tp + fp),
        'Recall': tp / (tp + fn),
        'Training Acc Score': train_score,
        'Validate Acc Score': validate_score,
        'Test Acc Score': test_score,
        'Acc Score Difference': validate_score - test_score
    }

    test_results = [eval_params]
    test_df = pd.DataFrame(test_results)
    return classifier, test_df