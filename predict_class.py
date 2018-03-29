import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import math
from decimal import Decimal
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import warnings
from sklearn import linear_model


def plot_roc(fpr, tpr, class_id):
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(int(class_id), roc_auc))


def random_forest(df, exclude_list):
    """
    Because we have an uneven class distribution, we want a fair classifier
    for all the classes in our dataset. We make a model for every class
    to be predicted
    """

    model_dict = dict()
    # First split dataset into all classes
    classes = df.Class.unique()

    # Divide dataset into four for cross-validation
    kf = KFold(n_splits=4)
    probability_dict = {u_class: list() for u_class in classes}

    for train_index, test_index in kf.split(df):
        dataset_train = df.iloc[train_index]
        dataset_test = df.iloc[test_index]
        for unique_class in classes:
            # print('Processing class: {}'.format(unique_class))
            # set all classes of unique class to 1, everything else is zero.
            dataset_train['current_class'] = 0
            dataset_test['current_class'] = 0
            dataset_train.loc[dataset_train['Class'] == unique_class, 'current_class'] = 1
            dataset_test.loc[dataset_test['Class'] == unique_class, 'current_class'] = 1

            # test max_depth
            forest = train_random_forest(dataset_train)

            # test model performance
            test_data = dataset_test.loc[dataset_test['current_class'] == 1]
            # Test the probabilities on test set, closer to 1 is better
            test_proba = (forest.predict_proba(test_data.drop(
                columns=['Class', 'current_class'])))
            avg_proba = sum([prob[1] for prob in test_proba])/len(test_proba)

            # print('Calculated probability: {}'.format(avg_proba))
            probability_dict[unique_class].append(avg_proba)

            # update dict with its model and score
            model_dict.update({unique_class: forest})

            whole_dataset = pd.concat([dataset_test, dataset_train])
            data_proba = (forest.predict_proba(whole_dataset.drop(
                columns=['Class', 'current_class'])))
            fpr, tpr, _ = roc_curve(whole_dataset.current_class, [prob[1] for prob in data_proba], pos_label=1)
            plot_roc(fpr, tpr, unique_class)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC plot SGD classifier')
        plt.legend(loc="lower right")
        plt.show()
        import sys; sys.exit()

    for u_class in model_dict.keys():
        print('{}'.format(
            str(sum(probability_dict[u_class])/len(probability_dict[u_class]))))


    import sys; sys.exit()
    # Test on the ultimate test set
    test_set = pd.read_csv('output_test_final.csv', index_col=0)
    test_set = test_set.drop(columns=exclude_list)
    test_set = test_set.fillna(0)
    for unique_class in classes:
        model = model_dict[unique_class]
        model_dict[unique_class] = model.predict_proba(test_set)

    # create output dataframe
    output_df = pd.DataFrame()
    output_df['ID'] = 0
    for unique_class in classes:
        output_df['class'+str(unique_class)] = 0

    # Calculate all probabilities
    for i in range(len(model_dict[classes[0]])):
        # Get the probabilities of each instance in the test set for each model
        probabilities = [model_dict[unique_class][i][1] for unique_class in classes]
        normalize = 1/sum(probabilities)
        probabilities = [prob*normalize for prob in probabilities]
        row = [str(i+1)] + probabilities
        output_df.loc[len(output_df)] = row

    output_df.to_csv('submission8.csv', index=False)


def train_random_forest(df):
    X = df.drop(columns=['Class', 'current_class'])
    y = df.current_class
    # forest = RandomForestClassifier(max_depth=50, random_state=0, n_estimators=100, bootstrap=False)
    forest = linear_model.SGDClassifier(loss='log')
    scores = cross_val_score(forest, X, y)
    # print(scores.mean())

    forest.fit(X, y)
    return forest


def main():
    warnings.filterwarnings("ignore")
    exclude_list = [
        'id',
        'Gene',
        'Variation',
        'most_common_word',
        'most_common_frequency',
        'amount_of_words',
        'from_start',
        'from_end'
    ]
    df = pd.read_csv('output.csv', index_col=0)
    df = df.drop(columns=exclude_list)
    df = df.fillna(0)

    random_forest(df, exclude_list)
    # train_test(df)


if __name__ == '__main__':
    main()
