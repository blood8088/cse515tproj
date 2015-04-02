class LabelParser:
	
	maxTax = 7
	taxDict = {
		0:'k',
		1:'p',
		2:'c',
		3:'o',
		4:'f',
		5:'g',
		6:'s' }
	
	def parseLabel(self,seq):
		labels = []
		import re
		for i in range(0,self.maxTax):
			match = re.search(self.taxDict[i]+"__\[?([A-Za-z0-9\-]+)\]?",seq)
			if match:
				labels.append(match.group(1))
			else:
				labels.append("")
		return labels