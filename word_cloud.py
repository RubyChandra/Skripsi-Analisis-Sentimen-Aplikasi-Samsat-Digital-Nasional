from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

tf_vector = CountVectorizer()

def cloudMaker(x, names):
	wc_mask = np.array(Image.open('static/image/logo_signal_mask.jpg'))
	freq_term=x.T.sum(axis=1)
	cloud = WordCloud(scale=2, background_color = "white", width=800, max_words=50,
					height=400, mask= wc_mask, colormap='Dark2').generate_from_frequencies(freq_term)
	cloud.to_file('static/image/wordcloud/cloud_'+names+'.png')
	freq_term = freq_term.sort_values(axis=0, ascending=False).head(20)
	# Determine the size of plot bar
	plt.figure(figsize=(14,7))
	ax = freq_term.plot.bar()
	# Placing label on top of each bars
	add_value_labels(ax)
	plt.savefig('static/image/frequency_bar_plot/plot_'+names+'.png')

def countFrequency(df, names):
	tf = tf_vector.fit_transform(df['processed_text'])
	features_names = tf_vector.get_feature_names_out()
	denses = tf.todense()
	lists = denses.tolist()
	term = pd.DataFrame(lists, columns = features_names)
	cloudMaker(term, names)

def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        # label = "{:.1f}".format(y_value)
        label = y_value

        # Create annotation
        ax.annotate(
           
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.
