class NaiveBayesianClassifier:

    def __init__(self,xTr,yTr):
        from numpy import array,where,amin
        self.n = yTr.shape[0]
        self.pis = ((array(xTr.sum(0))[0])+.5)/(self.n+1.0)
        self.labelset = [lbl for lbl in set(yTr) if lbl!='']
        self.indexset = [where(yTr==lbl)[0] for lbl in self.labelset]
        self.likelihood = array([(xTr[index].getnnz(0)+self.pis)/(len(index)+1) for index in self.indexset])

    def predict(self,xTe):
        from numpy import array,argmax
        from sympy.mpmath import fprod
        nPr = xTe.shape[0]
        yPr = [""]*xTe.shape[0]
        print('{}{}'.format("Number of classes: ",len(self.likelihood)))
        for i in range(nPr):
            x = xTe.getrow(i)
            likeli = self.likelihood[:,x.indices]          
            posterior = fprod(likeli.transpose())
            maxIndex = argmax(posterior)
            yPr[i]=self.labelset[maxIndex]
        return array(yPr)
        
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
