class NaiveBayesianClassifier:

    def __init__(self,xTr,yTr):
        from numpy import array,where
        self.n = yTr.shape[0]
        self.pis = ((array(xTr.sum(0))[0])+.5)/(self.n+1.0)
        self.labelset = [lbl for lbl in set(yTr) if lbl!='']
        self.indexset = [where(yTr==lbl)[0] for lbl in self.labelset]
        self.likelihood = [(xTr[index].getnnz(0)+self.pis)/(len(index)+1) for index in self.indexset]
        
    def predict(self,xTe):
        from numpy import array,float64
        nPr = xTe.shape[0]
        yPr = [""]*xTe.shape[0]
        print('{}{}'.format("Number of classes: ",len(self.likelihood)))
        for i in range(nPr):
            x = xTe.getrow(i)
            maxVal = float64(0.0)
            maxIndex = -1
            for j in range(len(self.likelihood)):
                posterior = float64(1.0)
                for pvg in self.likelihood[j][x.indices]:
                    posterior*=float64(pvg)
                if posterior>maxVal:
                    maxVal = posterior
                    maxIndex = j
            if maxIndex>-1:
                yPr[i]=self.labelset[maxIndex]
            else:
                yPr[i]=''
        return array(yPr)
        
    def accuracy(self,yTe,yPr):
        from numpy import float64
        if yTe.shape[0]==yPr.shape[0]:
            matchCount = 0.0
            totalCount = 0.0
            for i in range(yTe.shape[0]):
                if yTe[i]!='' and yPr[i]!='':
                    if yTe[i] == yPr[i]:
                        matchCount+=1.0
                    totalCount+=1.0
            return float64(matchCount)/totalCount
