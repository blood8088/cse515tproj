import labelParser

labelfp = open('..\\..\\rawData\\Gg_13_8_99.taxonomy\\gg_13_8_99.gg.tax')
labels = []

lblpsr = labelParser.LabelParser()

for line in labelfp:
	labels.append(lblpsr.parseLabel(line))

print(len(labels))
