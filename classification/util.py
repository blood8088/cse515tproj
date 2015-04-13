__author__ = 'Sui Jiang'

def rawDataPath(system):
    rawDataPath = 'rawData/Gg_13_8_99.taxonomy/'
    if system == 'Linux':
        rawDataPath = '/home/jglab/ypeng/projects/Qiezi_16s/' + rawDataPath
    else: rawDataPath = '../../' + rawDataPath
    return rawDataPath

def dataPath(system):
    dataPath = 'data/'
    if system == 'Linux':
        dataPath = '/home/jglab/ypeng/projects/Qiezi_16s/' + dataPath
    else: dataPath = '../../' + dataPath
    return dataPath


def save_feature_to_npz(file_name, x):
    from numpy import savez
    savez(file_name, data=x.data, indices=x.indices, indptr=x.indptr, shape=x.shape)
    
def load_feature_from_npz(feature_file_name):
    from numpy import load
    from scipy.sparse import csr_matrix
    feaFile = load(feature_file_name)
    features = csr_matrix((feaFile['data'],feaFile['indices'],feaFile['indptr']),feaFile['shape'])
    return features
    
def load_label_from_npz(label_file_name):
    from numpy import load
    lblFile = load(label_file_name)
    labels = lblFile['labels']
    return labels
    
