import pandas as pd
import openpyxl
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

#create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(kalimat):
	#remove number
	kalimat = re.sub(r'\d+', '', kalimat)
	#lowercase
	kalimat = kalimat.lower()
	#memsiahan huruf yang neempel koma
	kalimat = kalimat.replace(',', ' ')
	#remove punc
	kalimat = re.sub(r'[^\w\s]',' ', kalimat)
	# #remove double spaces
	kalimat = " ".join (kalimat.split())
	kalimat = re.sub (' +', ' ', kalimat)
	return kalimat

def removestopword(kalimat):
	stopwords = open('static/folder_kamus/kamus_stopwords.txt', 'r').read().split()
	content = ''
	filtered_text = [word for word in kalimat.split() if word not in stopwords]
	content = (" ".join(filtered_text))
	kalimat = content
	return kalimat

file_normalisasi = pd.read_excel('static/folder_kamus/kamus_normalisasi.xlsx')
kamus_normalisasi={}
for index,row in file_normalisasi.iterrows():
	if row[0] not in kamus_normalisasi:
		kamus_normalisasi[(row[0])]  = row[1]

def normalisasikata(kalimat):
	global normalisasi_kalimat
	normalisasi_kalimat = ''
	splited_kalimat = kalimat.split()
	for word in splited_kalimat:
		if word in kamus_normalisasi:
			word = kamus_normalisasi[word]
			normalisasi_kalimat = normalisasi_kalimat+' '+word
		else:
			word = word
			normalisasi_kalimat = normalisasi_kalimat+' '+word
	return normalisasi_kalimat