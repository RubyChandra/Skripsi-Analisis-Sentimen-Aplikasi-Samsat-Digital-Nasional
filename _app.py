from flask import Flask,render_template,redirect,url_for,make_response,request,flash,send_from_directory
from review_scraper import scraper_reviews
from parse_csv import parseCSV
from count_sentimen import countEachSentiment
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

app.config['UPLOAD_FOLDER'] = 'static/folder_csv'
app.config['DICTIONARY_FOLDER'] = "static/folder_kamus"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrap_review',methods=['POST','GET'])
def scrapReview():
    if request.method=='POST':
        inputJumlahKomentar = request.form['jumlah-komentar']
        files = scraper_reviews(int(inputJumlahKomentar))
        resp = make_response(files.to_csv(sep=';', index=False))
        resp.headers["Content-Disposition"] = "attachment; filename="+inputJumlahKomentar+" data_ulasan.csv"
        resp.headers["Content-Type"]  = "text/csv"
        return resp
    return render_template('scrap.html')

@app.route('/manage_dictionary')
def manageDictionary():
   return render_template('manage_dictionary.html', msg=False, success=False)

@app.route('/manage_dictionary/download_dict/<dict_name>')
def downloadDict(dict_name):
    return send_from_directory(directory=app.config["DICTIONARY_FOLDER"], path=dict_name, as_attachment=True)
       
@app.route('/manage_dictionary/update_stopwords',methods=['POST','GET'])
def updateStopDict():
   if request.method == 'POST':
      files_txt = request.files['file-upload']
      file_name,extension = files_txt.filename.split('.')
      if file_name == 'kamus_stopwords' and extension == 'txt':
         file_path = os.path.join(app.config['DICTIONARY_FOLDER'],files_txt.filename)
         files_txt.save(file_path)
         return render_template('manage_dictionary.html', msg=False, text= "File kamus_stopwords", success=True)
      else:
         return render_template('manage_dictionary.html', msg=True, text= "File kamus_stopwords", success=False)

@app.route('/manage_dictionary/update_normalize',methods=['POST','GET'])
def updateNormalizeDict():
   if request.method == 'POST':
      files_xlsx = request.files['file-upload']
      file_name,extension = files_xlsx.filename.split('.')
      if file_name == 'kamus_normalisasi' and extension == 'xlsx':
         file_path = os.path.join(app.config['DICTIONARY_FOLDER'],files_xlsx.filename)
         files_xlsx.save(file_path)
         return render_template('manage_dictionary.html', msg=False, text= "File kamus_normalisasi", success=True)
      else:
         return render_template('manage_dictionary.html', msg=True, text= "File kamus_normalisasi", success=False)

@app.route('/upload_review', methods=['POST','GET'])
def uploadReview():
    if request.method == 'POST':
        file_csv = request.files['file-upload']
        extension = file_csv.filename.split('.')[1]
        if file_csv.filename != '' and extension != 'csv':
            return render_template('upload_review.html', msg = True)
            # return "bukan file csv"
        else:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],file_csv.filename)
            # save files
            file_csv.save(file_path)
            # read files with pandas
            data_frame_labeled = parseCSV(file_path)
            if data_frame_labeled.columns[0] != "nama_pengguna" or data_frame_labeled['label'].isnull().any() == True:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'],file_csv.filename))
                return render_template('upload_review.html', msg = True)
            del data_frame_labeled['skor']
            del data_frame_labeled['tanggal']

            data_frame_labeled.insert(loc=2, column='Case Folding dan Cleansing', value='')
            data_frame_labeled.insert(loc=3, column='Tokenizing', value='')
            data_frame_labeled.insert(loc=4, column='Perbaikan Kata', value='')
            data_frame_labeled.insert(loc=5, column='Filtering', value='')
            data_frame_labeled.insert(loc=6, column='Stemming', value='')
            
            # Preprocessing
            for i in data_frame_labeled.index:    
                data_frame_labeled.at[i,'Case Folding dan Cleansing'] = prepro.caseandclean(
                    data_frame_labeled.at[i,'ulasan'])
                data_frame_labeled.at[i,'Tokenizing'] = data_frame_labeled.at[i,'Case Folding dan Cleansing'].split()
                data_frame_labeled.at[i,'Perbaikan Kata'] = prepro.normalisasikata(data_frame_labeled.at[i,'Case Folding dan Cleansing'])
                data_frame_labeled.at[i,'Filtering'] = prepro.removestopword((data_frame_labeled.at[i,'Perbaikan Kata']))
                data_frame_labeled.at[i,'Stemming'] = prepro.stemmer.stem(data_frame_labeled.at[i,'Filtering'])
                data_frame_labeled.at[i,'Stemming'] = prepro.normalisasikata(data_frame_labeled.at[i,'Stemming'])
                data_frame_labeled.at[i,'Stemming'] = prepro.removestopword(data_frame_labeled.at[i,'Stemming'])

            # incase there was a empty row after processing
            data_frame_labeled['Stemming'].replace('', np.nan, inplace=True)
            data_frame_labeled.dropna(subset = ['Stemming'], inplace=True)
        
            data_frame_temp = data_frame_labeled.copy()
            
            data_frame_temp.drop('nama_pengguna',inplace=True,axis=1)
            data_frame_temp.drop('ulasan',inplace=True,axis=1)
            data_frame_temp.drop('Case Folding dan Cleansing',inplace=True,axis=1)
            data_frame_temp.drop('Tokenizing',inplace=True,axis=1)
            data_frame_temp.drop('Perbaikan Kata',inplace=True,axis=1)
            data_frame_temp.drop('Filtering',inplace=True,axis=1)

            
            data_frame_temp.insert(loc=0, column='processed_text', value='')

            for i in data_frame_temp.index:
                data_frame_temp.at[i,'processed_text'] = (data_frame_temp.at[i,'Stemming'])
            del data_frame_temp['Stemming']
            
            data_frame_temp.to_csv('static/folder_csv/processed.csv',sep=';',index=None)

            return render_template('preprocessing.html',data1 = data_frame_labeled)


    return render_template('upload_review.html', msg = False)

@app.route('/model_training')
def modelTraining():
    data_frame_processed = parseCSV(filePath = "static/folder_csv/processed.csv")
    data_frame_processed.dropna(inplace=True)
    jlh_full = countEachSentiment(data_frame_processed) 

    # Lets say this for wordcloud and ploting bar
    df_pos = data_frame_processed.loc[data_frame_processed['label']==1]
    df_neg = data_frame_processed.loc[data_frame_processed['label']==-1]
    df_net = data_frame_processed.loc[data_frame_processed['label']==0]

    countFrequency(df_pos,'positif')
    countFrequency(df_neg,'negatif')
    countFrequency(df_net,'netral')
    # end of wordcloud

    dataset_X = data_frame_processed['processed_text']
    label_y = data_frame_processed['label']
    
    # Proses splitting dataset
    X_latih, X_uji, y_latih, y_uji = train_test_split(dataset_X,label_y, test_size=0.2, 
                                    random_state=42, shuffle=True, stratify=label_y)
    # Menghitung masing-masing jumlah data (training dan testing)
    df_latih = pd.DataFrame({'processed_text':X_latih.values,'label':y_latih.values})
    jlh_latih = countEachSentiment(df_latih)
    df_uji = pd.DataFrame({'processed_text':X_uji.values,'label':y_uji.values})
    jlh_uji = countEachSentiment(df_uji)
    # End of menghitung

    tfidf_vectorizer = TfidfVectorizer(smooth_idf=True)
    # Pembobotan TF_IDF
    X_latih_tfidf = tfidf_vectorizer.fit_transform(X_latih)
    X_uji_tfidf = tfidf_vectorizer.transform(X_uji)
    # end of pembobotan


    # Model Training  
    clasifier = MultinomialNB(alpha=0.01)
    clasifier.fit(X_latih_tfidf, y_latih)
    # prediksi
    y_pred = clasifier.predict(X_uji_tfidf)
    # y_pred = clasifier.predict(X_latih_tfidf)
    # end of prediksi

    # confusion matrix 
    # matrix_confusion = metrics.confusion_matrix(y_latih, y_pred)
    matrix_confusion = metrics.confusion_matrix(y_uji, y_pred)
    # accuracy
    # score_accuracy = metrics.accuracy_score(y_latih, y_pred)
    score_accuracy = metrics.accuracy_score(y_uji, y_pred)
    # precision
    # score_precision= metrics.precision_score(y_latih, y_pred, average=None, pos_label=1)
    score_precision= metrics.precision_score(y_uji, y_pred, average=None, pos_label=1)
    # recall
    # score_recall= metrics.recall_score(y_latih, y_pred, average=None, pos_label=1)
    score_recall= metrics.recall_score(y_uji, y_pred, average=None, pos_label=1)
    
    # end of klasifikasi

    # menyiapkan dataframe baru untuk menampilkan hasil prediksi dari data uji
    # data_frame_comparison = pd.DataFrame({'processed_text':X_latih.values,'label':y_latih.values, 'prediction':y_pred})
    data_frame_comparison = pd.DataFrame({'processed_text':X_uji.values,'label':y_uji.values, 'prediction':y_pred})

    return render_template('hasil_analisis.html', data = data_frame_comparison, jlh=[jlh_full,sum(jlh_full)],
                            jlh1=[jlh_latih,sum(jlh_latih)], jlh2=[jlh_uji,sum(jlh_uji)], skor =  score_accuracy,
                            confusion=matrix_confusion, skor_2= score_precision, skor_3 = score_recall)
    
if __name__ == '__main__':
    # Jangan lupa di hapus kalau sudah selesai
    app.run(debug=True)