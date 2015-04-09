import util,numpy
from util import load_feature_from_npz,load_label_from_npz
from labelParser import LabelParser
import nbclassifier,slicer

def nbclassify(iteration=1,lessData=False,test=False):
    print("Loading features and labels...")
    features = load_feature_from_npz(util.dataPath+'features.npz')
    labels = load_label_from_npz(util.dataPath+'labels.npz')
    print("Loading finished.")

    if lessData:
        portion = int(features.shape[0]/5)+1
        features = features[range(portion),:]
        labels = labels[range(portion),:]

    acc = numpy.zeros(LabelParser.maxTax)
    
    print("Starting classification...")
    slr = slicer.RandomSlicer(features.shape[0])
    print("Begin cutting data into training dataset and testing dataset...")
    xTr,yTr,xTe,yTe = slr.slice(features,labels)
    print("Cutting finished.")
    
    del features,labels

    if test:
        maxIter=1
    else:
        maxIter=LabelParser.maxTax

    for i in range(maxIter):
        print('{}{}'.format(i,"th taxonomy classification:"))
        print("Starting Training...")
        nbc = nbclassifier.NaiveBayesianClassifier(xTr,yTr[:,i])
        print("Training finished.")
        print("Starting Testing...")
        prediction = nbc.predict(xTe)
        print(prediction)
        print("Testing finished.")
        print("Starting calculating accuracy...")
        acc[i] += nbc.accuracy(yTe[:,i],prediction)
        print('{}{}{}'.format("Accuracy is ",acc[i],'.'))

    accDict = {}
    for i in range(LabelParser.maxTax):
        accDict[LabelParser.taxDict[i]] = acc[i]
    
    print(accDict)    
    numpy.savez(util.dataPath+"accuracy",accuracy = accDict)

if __name__ == '__main__':
    import sys
    if len(sys.argv)<2:
        nbclassify()
    elif len(sys.argv)==2:
        nbclassify(iteration=int(sys.argv[1]))
    elif len(sys.argv)==3:
        nbclassify(iteration=int(sys.argv[1]),lessData = (sys.argv[2]=='lessData'))
    elif len(sys.argv)==4:
        nbclassify(iteration=int(sys.argv[1]),lessData = (sys.argv[2]=='lessData'),test = (sys.argv[3]=='test'))
