import pandas
import numpy
import os
dirname = os.path.dirname(os.path.abspath(__file__))

dev_data_file = dirname + '/lol_dev.csv'
train_data_file = dirname + '/lol_train.csv'
test_data_file = dirname + '/lol_test.csv'

default_feature_set = ["side", "gdat10", "gdat15", "xpdat10", "fb", "firsttothreetowers", "result"]
post_15_feature_set = ["goldat15", "totalgold"]

def get_lcs_data(preparation_method="random", onlyUseBasicFeatures=False):
    which_feature_sets = [default_feature_set]

    if not onlyUseBasicFeatures:
        which_feature_sets.append(post_15_feature_set)

    #  Turning random off for now
    # "Random" means training sets and test and development datasets will be generated randomly. 
    if preparation_method == "random":
        # data = __preprocess_dataframe(__get_lcs_dataframe(), feature_sets=which_feature_sets)
        # return __split_data_random(data)
        return None
    # "Fixed" means we will use the 2017-2018 data as the training data, and 2019 data as the test data
    elif preparation_method == "fixed":
        training_df = __preprocess_dataframe(__get_lcs_dataframe(ignore=[test_data_file, dev_data_file]), feature_sets=which_feature_sets)
        dev_df = __preprocess_dataframe(__get_lcs_dataframe(ignore=[test_data_file, train_data_file]), feature_sets=which_feature_sets)
        test_df = __preprocess_dataframe(pandas.read_csv(test_data_file), feature_sets=which_feature_sets)
        return __split_df_and_label(training_df), __split_df_and_label(test_df), __split_df_and_label(dev_df)
    else:
        return None


"""
Filters and cleans the data and returns a dataframe.
"""
def __preprocess_dataframe(df=None, feature_sets=[default_feature_set, post_15_feature_set]):
    if df is not None:
        # Filters out only team stats
        df.loc[df['position'] == 'Team']

        # Filters, only keeping these features
        all_features = [feature for feature_set in feature_sets for feature in feature_set]
        df = df.filter(all_features)
        if post_15_feature_set in feature_sets:
            df["goldafter15"] = df["totalgold"] - df["goldat15"]
            df.drop(["goldat15", "totalgold"], axis=1, inplace=True)

        # Turn red and blue (side) into numbers -> Red=2 Blue=1
        df.replace(to_replace=["Blue", "Red"], value=[1, 2], inplace=True)

        # # Cleaning data - Keep only rows who do not have '#VALUE!' or '#DIV/0' in any of the remaining columns
        for column in df.columns:
            df = df.loc[~df[column].isin(['#VALUE!', '#DIV/0!'])]
            df = df.loc[~df[column].isnull()]

        print(df.columns)

    return df


""" 
Returns a pandas dataframe with all the LCS match data from each file. Each CSV file has the same column names, so
no issues should occur with concatenation. Will concat all csvs except for the ones to be ignored.
"""
def __get_lcs_dataframe(ignore=[]):
    csv_file_names = [
        dev_data_file,
        train_data_file,
        test_data_file]

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

    dev_df = training_df[~mask]
    training_df = training_df[mask]

    return __split_df_and_label(training_df), __split_df_and_label(test_df), __split_df_and_label(dev_df)


"""
Given a pandas dataframe, returns two dataframes - one with only the data, and one with only the corresponding labels.
No operations are to be performed after doing this split, so that order is preserved. 
Ex: returns (just_data_df, just_labels),
"""
def __split_df_and_label(df=None):
    just_data_df = df.drop(['result'], axis=1)
    just_labels_df = df.drop(df.columns.difference(['result']), axis=1)

    # normalize the values
    # x_normed = (x - x.min(0)) / x.ptp(0) from  https://stackoverflow.com/questions/29661574/normalize-numpy-array-columns-in-python

    just_data_numpy_array = just_data_df.values
    just_labels_numpy_array = just_labels_df.values

    normalized_data = (just_data_numpy_array  - just_data_numpy_array.min(0)) / just_data_numpy_array.ptp(0)

    return normalized_data, just_labels_numpy_array 
