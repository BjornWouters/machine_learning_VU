import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_data(data_file):
    df = pd.read_csv(data_file)
    return df


def plot_class_distribution(df):
    totals = []

    ax = df['Class'].value_counts().plot(kind='barh', figsize=(10, 7),
                                         fontsize=13)

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_width())

    # set individual bar lables using above list
    total = sum(totals)

    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width() + .3, i.get_y() + .10,
                str(round((i.get_width() / total) * 100, 2))
                + '%', fontsize=15, color='dimgrey')

    plt.xlabel('Occurrences')
    plt.ylabel('Cancer subtype')
    plt.title('Subtype distribution')


def plot_gene_distribution(df):
    df['Gene'].value_counts().plot(kind='bar', figsize=(10, 7),
                                        fontsize=5)

    plt.xlabel('Gene')
    plt.ylabel('Occurrences')
    plt.title('Gene mutation occurrences')
    plt.legend([df['Gene'].value_counts().head(10)], numpoints=10)


def plot_mutation_classification(df):
    df_freq = pd.DataFrame()
    for column in df:
        if column.startswith('M_'): # C_ will show the control
            series = df.loc[df[column] == 1]['Class'].value_counts()
            # series.rename(column)
            df_freq[column] = series

    sns.heatmap(df_freq, annot=True)
    plt.ylabel('Class')
    plt.title('Mutation distribution mutated')


def plot_mutation_position(df):
    df.loc[df['Class'] == 7].set_index('Class')\
        .from_start.dropna().sort_values().plot(kind='bar')


    plt.ylabel('Mutation position')
    plt.xlabel('Gene')
    plt.title('Mutation position (from start) on class 7')


def main():
    data_file = 'output.csv'
    df = read_data(data_file)
    # plot_class_distribution(df)
    # plot_gene_distribution(df)
    # plot_mutation_classification(df)
    plot_mutation_position(df)
    plt.show()


if __name__ == '__main__':
    main()
