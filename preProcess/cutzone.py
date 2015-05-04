import re
from util import rawDataPath
from platform import system

fprimer = ('','','','GTG[CT]CAGC[AC]GCCGCGGTAA')
rprimer = ('','','','ATTAGA[AT]ACCC[TCG][AGTC]GTAGTCC')

def cutZone(zone = 4):
    fp = open(rawDataPath(system())+'gg_13_8_99_sorted.fasta','r')
    nfp = open('{}{}{}'.format(rawDataPath(system())+'gg_13_8_99_v',zone,'.fasta'),'w')
    flag = 1
    count = 0
    print(zone)
    par = re.compile(fprimer[zone-1]+'([AGTC]+)'+rprimer[zone-1])
    for line in fp:
        if flag == 0:
            vseq = par.search(line)
            if vseq:
                nfp.write(vseq.group(1)+'\n')
            else: 
                count+=1
                nfp.write('\n')
            
        else:
            nfp.write(line)
        
        flag = (flag+1)%2
    nfp.close()
    print(count)

if __name__ == '__main__':
    import sys
    if(len(sys.argv)== 2):
        cutZone()
    if(len(sys.argv)== 3):
        cutZone(int(sys.argv[2]))
