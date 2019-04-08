import pandas
import numpy


all_2017_match_data_file = 'data/2017-match-data.csv'
spring_2018_match_data_file = 'data/2018-spring-match-data.csv'
summer_2018_match_data_file = 'data/2018-summer-match-data.csv'
worlds_2018_match_data_file = 'data/2018-worlds-match-data.csv'
spring_2019_match_data_file = 'data/2019-spring-match-data.csv'


def get_lcs_data(preparation_method="random"):
    # "Random" means training sets and test datasets will be generated randomly. 
    if preparation_method == "random":
        data = __preprocess_dataframe(__get_lcs_dataframe())
        return __split_data_random(data)
    # "Fixed" means we will use the 2017-2018 data as the training data, and 2019 data as the test data
    elif preparation_method == "fixed":
        training_df == __preprocess_dataframe(__get_lcs_dataframe(ignore=[spring_2019_match_data_file]))
        test_df = __preprocess_dataframe(pandas.read_csv(spring_2019_match_data_file))
        return __split_df_and_label(training_df), __split_df_and_label(test_df)
    else:
        return None

"""
Filters and cleans the data and returns a dataframe.
"""
def __preprocess_dataframe(df=None):
    if df is not None:
        # Filters out only team stats
        df.loc[df['position'] == 'Team']

        # Filters, only keeping these features
        df = df.filter(["side", "gdat10", "xpdat10", "gdat15", "fb", "firsttothreetowers", "result"])

        # Turn red and blue (side) into numbers -> Red=2 Blue=1
        df.replace(to_replace=["Blue", "Red"], value=[1, 2], inplace=True)

        # # Cleaning data - Keep only rows who do not have '#VALUE!' or '#DIV/0' in any of the remaining columns
        for column in df.columns:
            df = df.loc[~df[column].isin(['#VALUE!', '#DIV/0!'])]
            df = df.loc[~df[column].isnull()]

    return df



""" 
Returns a pandas dataframe with all the LCS match data from each file. Each CSV file has the same column names, so
no issues should occur with concatenation. Will concat all csvs except for the ones to be ignored.
"""
def __get_lcs_dataframe(ignore=[]):
    csv_file_names = [
        all_2017_match_data_file,
        spring_2018_match_data_file,
        summer_2018_match_data_file,
        worlds_2018_match_data_file,
        spring_2019_match_data_file]

    data = pandas.concat([pandas.read_csv(csv_file) for csv_file in csv_file_names if csv_file not in ignore])
    return data


"""
Given a pandas dataframe, randomly splits all the data into two separate data frames, with one containing 80% of the 
original data and the other containing 20% (i.e. the training data and test data). Returns a tuple of each dataframe
split into dataframe and labels. Ex: Returns ((training data, training labels), (test data, test labels))
"""
def __split_data_random(df=None):
    mask = numpy.random.rand(len(df)) < 0.8
    training_df = df[mask]
    test_df = df[~mask]

    return __split_df_and_label(training_df), __split_df_and_label(test_df)


"""
Given a pandas dataframe, returns two dataframes - one with only the data, and one with only the corresponding labels.
No operations are to be performed after doing this split, so that order is preserved. 
Ex: returns (just_data_df, just_labels),
"""
def __split_df_and_label(df=None):
    just_data_df = df.drop(['result'], axis=1)
    just_labels_df = df.drop(df.columns.difference(['result']), axis=1)

    return just_data_df.values, just_labels_df.values
