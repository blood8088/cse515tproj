__author__ = 'Sui Jiang'

rawDataPath = "..\\..\\rawData\\Gg_13_8_99.taxonomy\\"
dataPath = "..\\..\\data\\"


def save_data_to_npz(file_name, x):
    from numpy import savez
    savez(file_name, data=x.data, indices=x.indices, indptr=x.indptr, shape=x.shape)