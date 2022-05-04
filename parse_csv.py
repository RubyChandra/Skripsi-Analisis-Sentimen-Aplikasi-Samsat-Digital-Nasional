import pandas as pd

def parseCSV(filePath):
	dataframe = pd.read_csv(filePath, sep=';')
	return dataframe