class RandomSlicer:
    
    def __init__(self,up):
        self.upper = int(up)
        self.testnum = int(up)/5
        
    def updateUpper(self,up):
        self.upper = int(up)
        self.testnum = int(up)/5
    
    def slice(self,feature,label):
        from numpy.random import choice
        if label.shape[0] != self.upper:
            self.updateUpper(len(label))
        total = range(self.upper)
        test = choice(self.upper,self.testnum,replace=False)
        train = [index for index in total if index not in test]
        xTr = feature[train,:]
        yTr = label[train,:]
        xTe = feature[test,:]
        yTe = label[test,:]
        return (xTr,yTr,xTe,yTe)
