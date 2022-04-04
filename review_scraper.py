from google_play_scraper import reviews, Sort
from pandas import DataFrame

def scraper_reviews(jumlah):
	crawed_data,continuation_token = reviews(
		'app.signal.id',
		lang = 'id',
		country = 'ID',
		sort=Sort.NEWEST,
		count = jumlah
		)
	temp = []
	for i in range(len(crawed_data)):
		temp.append([
			crawed_data[i]['userName'],
			crawed_data[i]['at'],
			crawed_data[i]['score'],
			crawed_data[i]['content']
			])
	dataframe_ulasan  = DataFrame(temp, columns = ['nama_pengguna','tanggal','skor','ulasan'])
	dataframe_ulasan['label']  = ''

	return dataframe_ulasan