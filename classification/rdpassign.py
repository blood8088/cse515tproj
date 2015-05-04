import sys
from platform import system
if system()=='Linux':
    sys.path.append('/home/jglab/ypeng/software/anaconda3/lib/python3.4/')

import util,numpy
from scipy.sparse import csr_matrix
from util import load_feature_from_npz,load_label_from_npz
from labelParser import LabelParser
import classifier,slicer

def training_testing_split(iteration=1):
    print("Loading features and labels...")
    print('Data Path is: ' + util.dataPath(system())+ '.')
    features = load_feature_from_npz(util.dataPath(system())+'features.npz')
    labels = load_label_from_npz(util.dataPath(system())+'labels.npz')
    print("Loading finished.")

    slr = slicer.RandomSlicer(features.shape[0])
    print("Begin cutting data into training dataset and testing dataset...")
    xTr,yTr,xTe,yTe = slr.slice(features,labels)
    print("Cutting finished.")
    numpy.savez('{}{}{}'.format(util.dataPath(system()),'data_split_testing_',iteration), xTe_data = xTe.data, xTe_indices = xTe.indices, xTe_indptr = xTe.indptr, xTe_shape = xTe.shape, yTe = yTe)
    numpy.savez('{}{}{}'.format(util.dataPath(system()),'data_split_training_',iteration), xTr_data = xTr.data, xTr_indices = xTr.indices, xTr_indptr = xTr.indptr, xTr_shape = xTr.shape, yTr = yTr)


def load_training_testing(iteration=1):
    tr_data = numpy.load('{}{}{}'.format(util.dataPath(system())+'data_split_training_',iteration,'.npz'))
    te_data = numpy.load('{}{}{}'.format(util.dataPath(system())+'data_split_testing_',iteration,'.npz'))

    xTr = csr_matrix((tr_data['xTr_data'],tr_data['xTr_indices'],tr_data['xTr_indptr']),tr_data['xTr_shape'])
    yTr = tr_data['yTr']    
    xTe = csr_matrix((te_data['xTe_data'],te_data['xTe_indices'],te_data['xTe_indptr']),te_data['xTe_shape'])
    yTe = te_data['yTe'] 
    return xTr,yTr,xTe,yTe   

def nbclassify(iteration=1,test=False):

    if test:
        maxIter=1
    else:
        maxIter=LabelParser.maxTax

    acc = numpy.zeros(LabelParser.maxTax)
    yPr = []

    print('Loading data...')
    xTr,yTr,xTe,yTe = load_training_testing(iteration)   
 
    print("Starting classification...")

    for i in range(maxIter):
        print('{}{}'.format(i,"th taxonomy classification:"))
        print("Starting Training...")
        nbc = classifier.NaiveBayesianClassifier(xTr,yTr[:,i])
        print("Training finished.")
        print("Starting Testing...")
        prediction = nbc.predict(xTe)
        yPr.append(prediction)
        print(prediction)
        print("Testing finished.")
        print("Starting calculating accuracy...")
        acc[i] += nbc.accuracy(yTe[:,i],prediction)
        print('{}{}{}'.format("Accuracy is ",acc[i],'.'))

    yPr = numpy.array(yPr)
    yPr = yPr.transpose()

    print(acc)    
    numpy.savez('{}{}'.format(util.dataPath(system())+'nbcresult_',iteration),accuracy = acc,yPr = yPr)

def lrclassify(iteration=1,test=False):
     
    from sklearn.linear_model import LogisticRegression as LR

    if test:
        maxIter=1
    else:
        maxIter=LabelParser.maxTax

    acc = numpy.zeros(LabelParser.maxTax)
    yPr = []

    print('Loading data...')
    xTr,yTr,xTe,yTe = load_training_testing(iteration)   
 
    print("Starting classification...")
    lrc = LR()
    for i in range(maxIter):
        print('{}{}'.format(i,"th taxonomy classification:"))
        print("Starting Training...")
        lrc.fit(xTr,yTr[:,i])
        print("Training finished.")
        print("Starting Testing...")
        prediction = lrc.predict(xTe)
        yPr.append(prediction)
        print("Testing finished.")
        print("Starting calculating accuracy...")
        acc[i] += lrc.score(xTe,yTe[:,i],prediction)
        print('{}{}{}'.format("Accuracy is ",acc[i],'.'))

    yPr = numpy.array(yPr)
    yPr = yPr.transpose()

    print(acc)    
    numpy.savez('{}{}'.format(util.dataPath(system())+'lrcresult_',iteration),accuracy = acc,yPr = yPr)


if __name__ == '__main__':
    import sys
    if len(sys.argv)<2:
        raise Exception('Not enough arguments!')
    if sys.argv[1]=='split':
        if len(sys.argv)<3:
            training_testing_split()
        elif len(sys.argv)==3:
            training_testing_split(int(sys.argv[2]))
        else:
            raise Exception('Error!Too many arguments!')
    elif sys.argv[1]=='nbclassify':
        if len(sys.argv)<3:
            nbclassify()
        elif len(sys.argv)==3:
            nbclassify(iteration=int(sys.argv[2]))
        elif len(sys.argv)==4:
            nbclassify(iteration=int(sys.argv[2]),test = (sys.argv[3]=='test'))
        else:
            raise Exception('Error!Too many arguments!')
    elif sys.argv[1]=='lrclassify':
        if len(sys.argv)<3:
            lrclassify()
        elif len(sys.argv)==3:
            lrclassify(iteration=int(sys.argv[2]))
        elif len(sys.argv)==4:
            lrclassify(iteration=int(sys.argv[2]),test = (sys.argv[3]=='test'))
        else:
            raise Exception('Error!Too many arguments!')
    else:
        raise Exception('Wrong arguments!Try nbclassify, lrclassify or split!')
