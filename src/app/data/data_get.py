import pandas
import numpy


def get_lcs_data():
    spring_2018_match_data_file = 'data/2018-spring-match-data.csv'
    summer_2018_match_data_file = 'data/2018-summer-match-data.csv'
    worlds_2018_match_data_file = 'data/2018-worlds-match-data.csv'
    csv_file_names = [spring_2018_match_data_file, summer_2018_match_data_file, worlds_2018_match_data_file]
    data = pandas.concat([pandas.read_csv(csv_file) for csv_file in csv_file_names])

    # Filters out only team stats
    data = data.loc[data['position'] == 'Team']

    data = data.filter(["side", "gdat10", "xpdat10", "gdat15", "fb", "firsttothreetowers", "result"])

    # Turn red and blue (side) into numbers -> Red=2 Blue=1
    data.replace(to_replace=["Blue", "Red"], value=[1, 2], inplace=True)

    # # Cleaning data - Keep only rows who do not have '#VALUE!' or '#DIV/0' in any of the remaining columns
    for column in data.columns:
        data = data.loc[~data[column].isin(['#VALUE!', '#DIV/0!'])]
        data = data.loc[~data[column].isnull()]

    return __split_data(data)


def __split_data(df=None):
    mask = numpy.random.rand(len(df)) < 0.8
    training_df = df[mask]
    test_df = df[~mask]

    return __split_df_and_label(training_df), __split_df_and_label(test_df)


def __split_df_and_label(df=None):
    just_data_df = df.drop(['result'], axis=1)
    just_labels_df = df.drop(df.columns.difference(['result']), axis=1)

    return just_data_df.values, just_labels_df.values
