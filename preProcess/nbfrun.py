import util,numpy,scipy
from util import load_feature_from_npz,load_label_from_npz
from labelParser import LabelParser
import nbclassifier,slicer
import multiprocessing as mp

maxIter = 1

print("Loading features and labels...\n")
features = load_feature_from_npz(util.dataPath+'features.npz')
labels = load_label_from_npz(util.dataPath+'labels.npz')
print("Loading finished.\n")

acc = numpy.zeros(LabelParser.maxTax)

print("Starting classification...\n")
for j in range(maxIter):
    print('{}{}'.format(j,"th iteration begins.\n"))
    slr = slicer.RandomSlicer(features.shape[0])
    print("Begin cutting data into training dataset and testing dataset...\n")
    xTr,yTr,xTe,yTe = slr.slice(features,labels)
    print("Cutting finished.\n")
    for i in range(2):
        print('{}{}'.format(i,"th taxonomy classification:\n"))
        print("Starting Training...\n")
        nbc = nbclassifier.NaiveBayesianClassifier(xTr,yTr[:,i])
        print("Training finished.\n")
        print("Starting Testing...\n")
        yPr = nbc.predict(xTe)
        print("Testing finished.\n")
        print("Starting calculating accuracy...\n")
        acc[i] += nbc.accuracy(yTe[:,i],yPr)
        print("Accuracy calculated.\n")
    print('{}{}'.format(j,"th iteration finished.\n"))
print("All classification iterations have finished.\n")

acc = acc/maxIter
accDict = {}
for i in range(LabelParser.maxTax):
    accDict[LabelParser.taxDict[i]] = acc[i]
    
print(accDict)
        
numpy.savez(util.dataPath+"accuracy_10iter",accuracy = accDict)
