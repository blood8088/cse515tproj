import rpStrHash

class FeaParser:
	
    def __init__(self):
        self.feaLen = int((4**8)/2)
        self.fea =[]

    def parse(self,seq,idn=None):
        self.fea=[0]*self.feaLen
        rp = rpStrHash.RPStrHash(seq)
        self.fea[rp.hVal]+=1
        for c in seq[8:]:
            rp.delLeft()
            rp.addRight(c)
            self.fea[rp.hVal]+=1
        if idn:
            self.fea[-1]=idn
        return self.fea
    
    
        
