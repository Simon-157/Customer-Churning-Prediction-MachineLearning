from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def compute_predictions_dataframe(explore_data, classifier, x_test):
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

    churn_proba = classifier.predict_proba(x_test)
    probabilities = DataFrame(churn_proba, columns = ['probability_not_churned', 'probability_churned'])
    reset_explore_data = explore_data.reset_index()
    reset_explore_data['probability_not_churned'] = probabilities['probability_not_churned']
    reset_explore_data['probability_churned'] = probabilities['probability_churned']


    predictions = classifier.predict(x_test)
    reset_explore_data['predicted'] = predictions


    results_dataframe = reset_explore_data[['customerID', 'probability_churned', 'probability_not_churned', 'predicted']]
    return results_dataframe


def test_classifier(x_train, y_train, X_validate, y_validate, x_test, y_test):
    classifier = RandomForestClassifier(random_state = 123, max_depth = 10, min_samples_leaf = 5)

    classifier.fit(x_train, y_train)

    train_score = classifier.score(x_train, y_train)

    validate_score = classifier.score(X_validate, y_validate)

    test_score = classifier.score(x_test, y_test)

    classifier_predictions = classifier.predict(x_test)

    tp = confusion_matrix(y_test, classifier_predictions)[1][1]
    fp = confusion_matrix(y_test, classifier_predictions)[0][1]
    tn = confusion_matrix(y_test, classifier_predictions)[0][0]
    fn = confusion_matrix(y_test, classifier_predictions)[1][0]


    eval_params = {
        'max_depth':10,
        'min_samples_leaf': 5,
        'True Positves': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatvies': fn,
        'Recall': tp / (tp + fn),
        'Training Score': train_score,
        'Precision': tp / (tp + fp),
        'Validate Score': validate_score,
        'Test Score': test_score,
        'Score Difference': validate_score - test_score
    }

    test_results_df = DataFrame([eval_params])

    return classifier, test_results_df


def write_to_csv(data_df):
    """
    It takes a dataframe as input and writes it to a csv file
    :param data_df: The dataframe that contains the data to be written to the CSV file
    """
    data_df.to_csv("test-results.csv")