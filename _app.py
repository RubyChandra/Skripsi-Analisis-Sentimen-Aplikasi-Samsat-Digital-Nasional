from flask import Flask,render_template,redirect,url_for,make_response,request
from review_scraper import scraper_reviews
from parse_csv import parseCSV
from count_sentimen import countEachSentiment,separateSentiment
import preprocessing as prepro
import pandas as pd
import numpy as np
import os

# sklearn library
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

from word_cloud import countFrequency

app = Flask(__name__)

UPLOAD_FOLDER = 'static/folder_csv'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global temp_dataframe

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrap_review',methods=['POST','GET'])
def scrapReview():
    if request.method=='POST':
        inputJumlahKomentar = request.form['jumlah-komentar']
        files = scraper_reviews(int(inputJumlahKomentar))
        resp = make_response(files.to_csv(sep=';', index=False))
        resp.headers["Content-Disposition"] = "attachment; filename=data_ulasan.csv"
        resp.headers["Content-Type"]  = "text/csv"
        return resp
    return render_template('scrap.html')

@app.route('/upload_review', methods=['POST','GET'])
def uploadReview():
    if request.method == 'POST':
        file_csv = request.files['file-upload']
        if file_csv.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],file_csv.filename)
            # save files
            file_csv.save(file_path)
            # read files with pandas
            data_frame_labeled = parseCSV(file_path)
            del data_frame_labeled['skor']
            del data_frame_labeled['tanggal']

            data_frame_labeled.insert(loc=2, column='Case Folding dan Cleansing', value='')
            data_frame_labeled.insert(loc=3, column='Tokenizing', value='')
            data_frame_labeled.insert(loc=4, column='Perbaikan Kata', value='')
            data_frame_labeled.insert(loc=5, column='Filtering', value='')
            data_frame_labeled.insert(loc=6, column='Stemming', value='')
            # Preprocessing
            for i in data_frame_labeled.index:
                data_frame_labeled.at[i,'Case Folding dan Cleansing'] = prepro.preprocess(
                    data_frame_labeled.at[i,'ulasan'])
            data_frame_labeled['Case Folding dan Cleansing'].replace('',np.nan,inplace=True)
            data_frame_labeled.dropna(subset=['Case Folding dan Cleansing'], inplace=True)
            data_frame_labeled.reset_index()

            for i in data_frame_labeled.index:    
                data_frame_labeled.at[i,'Tokenizing'] = data_frame_labeled.at[i,'Case Folding dan Cleansing'].split()
                data_frame_labeled.at[i,'Perbaikan Kata'] = prepro.normalisasikata(data_frame_labeled.at[i,'Case Folding dan Cleansing']).split()
                data_frame_labeled.at[i,'Filtering'] = prepro.removestopword(' '.join(data_frame_labeled.at[i,'Perbaikan Kata'])).split()
                data_frame_labeled.at[i,'Stemming'] = (' '.join(data_frame_labeled.at[i,'Filtering']))
                data_frame_labeled.at[i,'Stemming'] = prepro.stemmer.stem(data_frame_labeled.at[i,'Stemming'])
                data_frame_labeled.at[i,'Stemming'] = prepro.removestopword(data_frame_labeled.at[i,'Stemming']).split()
        
            data_frame_temp = data_frame_labeled.copy()
            del data_frame_temp['Tokenizing']
            del data_frame_temp['Case Folding dan Cleansing']
            del data_frame_temp['ulasan']
            del data_frame_temp['nama_pengguna']
            del data_frame_temp['Perbaikan Kata']
            del data_frame_temp['Filtering']
            data_frame_temp.insert(loc=0, column='processed_text', value='')
            for i in data_frame_temp.index:
                data_frame_temp.at[i,'processed_text'] = ' '.join(data_frame_temp.at[i,'Stemming'])
            del data_frame_temp['Stemming']
            
            data_frame_temp.to_csv('static/folder_csv/processed.csv',sep=';',index=None)

            return render_template('preprocessing.html',data1 = data_frame_labeled)

    return render_template('upload_review.html')

@app.route('/model_training')
def modelTraining():
    data_frame_processed = parseCSV(filePath = "static/folder_csv/processed.csv")
    data_frame_processed.dropna(inplace=True)
    jlh_full = countEachSentiment(data_frame_processed) 

    df_pos = separateSentiment(data_frame_processed,1)
    df_neg = separateSentiment(data_frame_processed,-1)
    df_net = separateSentiment(data_frame_processed,0)

    # Lets say this for wordcloud
    countFrequency(df_pos,'cloud_positif')
    countFrequency(df_neg,'cloud_negatif')
    countFrequency(df_net,'cloud_netral')
    # end of wordcloud

    train_X = data_frame_processed['processed_text']
    train_y = data_frame_processed['label']
    
    X_train, X_test, y_train, y_test = train_test_split(train_X,train_y, test_size=0.2, random_state=42)
    # Menghitung masing-masing jumlah data (training dan testing)
    df_train = pd.DataFrame({'processed_text':X_train.values,'label':y_train.values})
    jlh_train = countEachSentiment(df_train)
    df_test = pd.DataFrame({'processed_text':X_test.values,'label':y_test.values})
    jlh_test = countEachSentiment(df_test)
    # End of menghitung

    # Klasifikasi
    clasifier = MultinomialNB(alpha=1)
    tfidf_vectorizer = TfidfVectorizer()
    # Pembobotan TF_IDF
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    # end of pembobotan
    clasifier.fit(X_train_tfidf, y_train)
    # prediksi
    y_pred = clasifier.predict(X_test_tfidf)
    # end of prediksi

    # accuracy
    score_accuracy = metrics.accuracy_score(y_test, y_pred)
    # confusion matrix 
    matrix_confusion = metrics.confusion_matrix(y_test, y_pred)
    # end of klasifikasi
    data_frame_prediction  = pd.DataFrame({'processed_text':X_test.values,'label':y_test.values, 'prediction':y_pred})
    return render_template('hasil_analisis.html', data = data_frame_prediction, jlh=[jlh_full,sum(jlh_full)],
                            jlh1=[jlh_train,sum(jlh_train)], jlh2=[jlh_test,sum(jlh_test)], skor = score_accuracy,
                            confusion=matrix_confusion)

if __name__ == '__main__':
    # Jangan lupa di hapus kalau sudah selesai
    app.run(debug=True)