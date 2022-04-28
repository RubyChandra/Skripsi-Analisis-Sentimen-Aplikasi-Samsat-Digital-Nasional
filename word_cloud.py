from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from PIL import Image

tf_vector = CountVectorizer()

def cloudMaker(x, names):
	wc_mask = np.array(Image.open('static/image/logo_signal_mask.jpg'))
	cloud = WordCloud(scale=2, background_color = "white", max_words=50,
			width=800, height=400, mask= wc_mask).generate_from_frequencies(
			x.T.sum(axis=1))
	cloud.to_file('static/image/wordcloud/'+names+'.png')

def countFrequency(df, names):
	tf = tf_vector.fit_transform(df['processed_text'])
	features_names = tf_vector.get_feature_names_out()
	denses = tf.todense()
	lists = denses.tolist()
	term = pd.DataFrame(lists, columns = features_names)
	cloudMaker(term, names)