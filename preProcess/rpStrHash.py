class RPStrHash:
    
    cDict = {
            'a':0,
            'A':0,
            'c':1,
            'C':1,
            'g':2,
            'G':2,
            't':3,
            'T':3 }

    def __init__(self,seq):
        self.strLen = 8
        self.hVal = 0
        self.oVal = 0
        self.modVal = 4**7
        self.maxVal = 4**8-1
        if len(seq)>=8:
            self.sHash(seq[0:8])
	
    def cHash(self,char):
        return self.cDict[char]

    def delLeft(self):
        self.oVal = self.oVal%self.modVal

    def addRight(self,char):
        if len(char)==1:
            self.oVal = self.oVal*4+self.cHash(char)
            self.hVal = min(self.oVal,self.maxVal-self.oVal)
            return self.hVal

    def sHash(self,seq):
        if len(seq)==self.strLen:
            for c in seq:
                self.oVal = self.oVal*4+self.cHash(c)
            self.hVal = min(self.oVal,self.maxVal-self.oVal)

            
