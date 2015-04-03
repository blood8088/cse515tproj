import util,numpy,scipy
from util import load_feature_from_npz,load_label_from_npz
import nbclassifier,slicer

features = load_feature_from_npz(util.dataPath+'features.npz')
labels = load_label_from_npz(util.dataPath+'labels.npz')

slr = slicer.RandomSlicer(features.shape[0])
xTr,yTr,xTe,yTe = slr.slice(features,labels)

nbc = nbclassifier.NaiveBayesianClassifier(xTr,yTr[:,0])
yPr = nbc.predict(xTe)
acc = nbc.accuracy(yTe[:,0],yPr)

print(acc)
