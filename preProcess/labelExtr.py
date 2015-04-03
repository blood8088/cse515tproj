import labelParser,util,numpy

labelfp = open(util.rawDataPath+'gg_13_8_99.gg.tax')
labels = []

lblpsr = labelParser.LabelParser()

for line in labelfp:
	labels.append(lblpsr.parseLabel(line))

numpy.savez(util.dataPath+"labels",labels=numpy.array(labels))
