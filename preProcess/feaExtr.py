import feaParser,re,csv,numpy,util
from scipy.sparse import csr_matrix
from util import save_feature_to_npz

data=[]
row=[]
col=[]


feaFp = open(util.rawDataPath+'gg_13_8_99_sorted.fasta')
flag = 0

seqPat = re.compile(r'[AaCcGgTt]+')

parser = feaParser.FeaParser()
fea = []
i=0

for line in feaFp:
    if flag==1:
        seq = seqPat.search(line).group()
        fea = numpy.array(parser.parse(seq))
        ind = numpy.nonzero(fea)[0]
        col.extend(ind)
        row.extend(numpy.full(ind.shape,i))
        data.extend(fea[ind])
        i+=1
        
    flag = (flag+1)%2


row = numpy.array(row,dtype=numpy.int32)
col = numpy.array(col,dtype=numpy.uint16)
data = numpy.array(data,dtype=numpy.float32)

save_feature_to_npz(util.dataPath+"features", csr_matrix((data, (row, col))))
