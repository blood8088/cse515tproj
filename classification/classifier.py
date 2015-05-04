
class Classifier:

    def __init__(self):
        raise Exception('Not Implemented Yet!')
    
    def fit(self,xTr=xTr,yTr=yTr,xVa=xVa,yVa=yVa):
        raise Exception('Not Implemented Yet!')

    def predict(self,xTe):
        raise Exception('Not Implemented Yet!')

    def accuracy(self,yTe,yPr):
        raise Exception('Not Implemented Yet!')


class NaiveBayesianClassifier(Classifier):

    def __init__(self):
        from numpy import array
        self.nTr = 0
        self.pis = array([])
        self.labelset = []
        self.indexset = []
        self.likelihood = []
	self.threshold = 0.8

    def fit(self,xTr=xTr,yTr=yTr,xVa=xVa,yVa=yVa):
        from numpy import array,where,amin,log
        self.nTr = yTr.shape[0]
        self.pis = ((array(xTr.sum(0))[0])+.5)/(self.n+1.0)
        self.labelset = [lbl for lbl in set(yTr) if lbl!='']
        self.indexset = [where(yTr==lbl)[0] for lbl in self.labelset]
        self.likelihood = log([(xTr[index].getnnz(0)+self.pis)/(len(index)+1) for index in self.indexset])
        if xVa and yVa:
            self.conf_th = 1
            nVa = xVa.shape[0]
            for i in range(nVa):
                x = xVa.getrow(i)
                y = yVa[i]
                self.conf_th = min(self.conf_th,self.confidence_threshold(x,y))

    def predict(self,xTe):
        nPr = xTe.shape[0]
        yPr = self.get_label(xTe)
        for i in range(nPr):
            x = xTe.getrow(i)
            y = yPr[i]
            if(self.confidence_threshold(x,y)<0.8):
                yPr[i] = ''
        return yPr
        
 
    def get_label(self,xTe):
        from numpy import array,argmax,sum
        nPr = xTe.shape[0]
        yPr = ['']*xTe.shape[0]
        for i in range(nPr):
            x = xTe.getrow(i)
            likeli = self.likelihood[:,x.indices]          
            posterior = sum(likeli,axis = 1)
            maxIndex = argmax(posterior)
            yPr[i]=self.labelset[maxIndex]
        return array(yPr)

    def confidence_threshold(self,xte,label):
        from numpy.random import choice
        from numpy import array,argmax,sum
        n = len(xte.indices)/8
        ypr = []
        for i in range(100):
            indi = choice(xte.indices,n)
            likeli = self.likelihood[:,indi]
            poste = sum(likeli,axis = 1)
            maxIn = argmax(poste)
            ypr.append(self.labelset[maxIn])
        cnt = ypr.count(label)
        return cnt/100.0

    def accuracy(self,yTe,yPr):
        from numpy import float32
        if yTe.shape[0]==yPr.shape[0]:
            matchCount = 0.0
            totalCount = 0.0
            for i in range(yTe.shape[0]):
                if yTe[i]!='' and yPr[i]!='':
                    if yTe[i] == yPr[i]:
                        matchCount+=1.0
                    totalCount+=1.0
            return float32(matchCount)/totalCount
