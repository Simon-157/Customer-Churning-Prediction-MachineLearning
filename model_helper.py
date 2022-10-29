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
    model_dicts = []


    for num in range(2, 11):
        for val in range(1, 20):
            classifier = RandomForestClassifier(random_state = 123, max_depth = num, min_samples_leaf = val)

            classifier.fit(x_train, y_train)


            train_score = classifier.score(x_train, y_train)

            predictions = classifier.predict(x_validate)

            #Use confusion matrix to find TP, FP, TN, FN
            tp = confusion_matrix(y_validate, predictions)[1][1]
            fp = confusion_matrix(y_validate, predictions)[0][1]
            tn = confusion_matrix(y_validate, predictions)[0][0]
            fn = confusion_matrix(y_validate, predictions)[1][0]
            #Score the model on validate data
            validate_score = classifier.score(x_validate, y_validate)

            #Create a dictionary for model values
            output = {
                'max_depth':num,
                'min_samples_leaf': val,
                'True Positves': tp,
                'False Positives': fp,
                'True Negatives': tn,
                'False Negatvies': fn,
                'Precision': tp / (tp + fp),
                'Recall': tp / (tp + fn),
                'Training Score': train_score,
                'Validate Score': validate_score,
                'Score Difference': train_score - validate_score
            }

            model_dicts.append(output)

    return pd.DataFrame(model_dicts)


def test_classifier(x_train, y_train, X_validate, y_validate, x_test, y_test):

    classifier = RandomForestClassifier(random_state = 123, max_depth = 10, min_samples_leaf = 5)

    classifier.fit(x_train, y_train)

    train_score = classifier.score(x_train, y_train)

    validate_score = classifier.score(X_validate, y_validate)

    test_score = classifier.score(x_test, y_test)

    clf_preds = classifier.predict(x_test)

    tp = confusion_matrix(y_test, clf_preds)[1][1]
    fp = confusion_matrix(y_test, clf_preds)[0][1]
    tn = confusion_matrix(y_test, clf_preds)[0][0]
    fn = confusion_matrix(y_test, clf_preds)[1][0]


    eval_params = {
        'max_depth':10,
        'min_samples_leaf': 5,
        'True Positves': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatvies': fn,
        'Precision': tp / (tp + fp),
        'Recall': tp / (tp + fn),
        'Training Score': train_score,
        'Validate Score': validate_score,
        'Test Score': test_score,
        'Score Difference': validate_score - test_score
    }

    test_results = [eval_params]

    test_df = pd.DataFrame(test_results)

    return classifier, test_df


def compute_predictions_dataframe(explore_data, classifier, X_test):
    """
    The function takes in the explore data, the classifier, and the X_test data, and returns a dataframe
    with the customerId, the probability of churn, the probability of not churning, and the predicted
    churn value

    :param explore_data: the dataframe that contains the customerId and the target variable
    :param classifier: the model you want to use to make predictions
    :param X_test: the test dataframe
    :return: A dataframe with the customerId, probability_churned, probability_not_churned, and
    predicted columns.
    """

    churn_proba = classifier.predict_proba(X_test)
    probabilities = pd.DataFrame(churn_proba, columns = ['probability_not_churned', 'probability_churned'])
    reset_explore_data = explore_data.reset_index()
    reset_explore_data['probability_not_churned'] = probabilities['probability_not_churned']
    reset_explore_data['probability_churned'] = probabilities['probability_churned']


    predictions = classifier.predict(X_test)
    reset_explore_data['predicted'] = predictions


    results_dataframe = reset_explore_data[['customerId', 'probability_churned', 'probability_not_churned', 'predicted']]

    return results_dataframe