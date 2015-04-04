class NaiveBayesianClassifier:

    def __init__(self,xTr,yTr):
        from numpy import array,where
        self.n = len(yTr)
        self.pis = ((array(xTr.sum(0))[0])+.5)/(self.n+1.0)
        self.labelset = [lbl for lbl in set(yTr)]
        self.indexset = [where(yTr==lbl) for lbl in self.labelset]
        self.likelihood = [(xTr[index].getnnz(0)+self.pis)/(len(index)+1) for index in self.indexset]
        
        
    def predict(self,xTe):
        from numpy import array,float64
        nPr = xTe.shape[0]
        yPr = [""]*xTe.shape[0]
        print('{}{}'.format("Number of classes: ",len(self.likelihood)))
        for i in range(nPr):
            x = xTe.getrow(i)
            maxVal = 0
            maxIndex = 0
            for j in range(len(self.likelihood)):
                posterior = float64(1.0)
                for pvg in self.likelihood[j][x.indices]:
                    posterior*=float64(pvg)
                if posterior>maxVal:
                    maxVal = posterior
                    maxIndex = j
            yPr[i]=self.labelset[maxIndex]
        return array(yPr)
        
    def accuracy(self,yTe,yPr):
        from numpy import float64
        if yTe.shape[0]==yPr.shape[0]:
            count = 0.0
            for i in range(yTe.shape[0]):
                if yTe[i] == yPr[i]:
                    count+=1.0
            return float64(count)/yTe.shape[0]