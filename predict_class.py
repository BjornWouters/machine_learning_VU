import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import math
from decimal import Decimal
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def random_forest(df):
    """
    Because we have an uneven class distribution, we want a fair classifier
    for all the classes in our dataset. We make a model for every class
    to be predicted
    """

    model_dict = dict()
    # First split dataset into all classes
    classes = df.Class.unique()
    for unique_class in classes:
        print('Processing class: {}'.format(unique_class))
        # set all classes of unique class to 1, everything else is zero.
        df['current_class'] = 0
        df.loc[df['Class'] == unique_class, 'current_class'] = 1
        forest = train_random_forest(df)

        # update dict with its model and score
        model_dict.update({unique_class: forest})

    # Test on the ultimate test set
    test_set = pd.read_csv('output_test.csv', index_col=0)
    exclude_list = [
        'ID',
        'Gene',
        'Variation',
        'from_start',
        'from_end'
    ]
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

    output_df.to_csv('submission.csv', index=False)


def train_random_forest(df):
    X = df.drop(columns=['Class', 'current_class'])
    y = df.current_class
    forest = RandomForestClassifier(max_depth=2, random_state=0)

    scores = cross_val_score(forest, X, y)
    print(scores.mean())

    forest.fit(X, y)
    return forest


def main():
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

    random_forest(df)
    # train_test(df)


if __name__ == '__main__':
    main()
