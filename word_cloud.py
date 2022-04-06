from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt

tf_vector = TfidfVectorizer()

def cloudMaker(x, names):
	cloud = WordCloud(background_color = "white", max_words=35, width=800, height=400).generate_from_frequencies(
			x.T.sum(axis=1))
	plt.figure(figsize=(20,10))
	plt.imshow(cloud, interpolation="bilinear")
	plt.axis("off")
	plt.tight_layout(pad=0)
	plt.savefig('static/image/wordcloud/'+names+'.png')

def countFrequency(df, names):
	tf = tf_vector.fit_transform(df['processed_text'])
	features_names = tf_vector.get_feature_names_out()
	denses = tf.todense()
	lists = denses.tolist()
	term = pd.DataFrame(lists, columns = features_names)
	cloudMaker(term, names)