import pandas as pd

def parseCSV(filePath):
	dataframe = pd.read_csv(filePath, sep=';', encoding = "ISO-8859-1")
	return dataframe