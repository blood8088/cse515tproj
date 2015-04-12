import sys
sys.path.append('/home/jglab/ypeng/software/anaconda3/lib/python3.4/')
from platform import system
import util,numpy
from util import load_feature_from_npz,load_label_from_npz
from labelParser import LabelParser
import classifier,slicer

def nbclassify(iteration=1,lessData=False,test=False):
    print("Loading features and labels...")
    features = load_feature_from_npz(util.dataPath(system())+'features.npz')
    labels = load_label_from_npz(util.dataPath(system())+'labels.npz')
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

    yPr = []

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
    numpy.savez('{}{}'.format(util.dataPath(system())+'nbcresult_',iteration),accuracy = acc,yPr = yPr, yTe = yTe)

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
