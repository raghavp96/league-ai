from .data_get import get_lcs_data

def get_data(preparation_method="random", onlyUseBasicFeatures=False):
    return get_lcs_data(preparation_method, onlyUseBasicFeatures=onlyUseBasicFeatures)