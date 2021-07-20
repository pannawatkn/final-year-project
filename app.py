# '''''''''''''''''''''''''''''''''''
# '''''''''''' IMPORT '''''''''''''''
# '''''''''''''''''''''''''''''''''''
# flask
from flask import Flask, request, render_template, session

# pygal
import pygal

# keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# nltk
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

# Utility
import pickle
import string
import re
import json
import tweepy
import numpy as np

# DataFrame
import pandas as pd

# Matplot
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

# Config
import config


app = Flask(__name__)
app.secret_key = config.secret_key

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load Model (Information Extraction)
vec = pickle.load(open('classification-model/dictVectorizer.pickle', 'rb'))
rf_model = pickle.load(open('classification-model/rf.pickle', 'rb'))

# Load Model (Sentiment Analysis)
word2idx = pickle.load(open('deep-learning-model/word2idx.pickle', 'rb'))
idx2word = pickle.load(open('deep-learning-model/idx2word.pickle', 'rb'))
target2idx = pickle.load(open('deep-learning-model/target2idx.pickle', 'rb'))
idx2target = pickle.load(open('deep-learning-model/idx2target.pickle', 'rb'))
model = load_model(
    'deep-learning-model/bi_gru_model_10_epoch_general_word_embedding.h5')

# Home Page
@app.route('/')
def home_page():
    return render_template('home_page.html')


# Information Extraction Page
@app.route('/information_extraction')
def ie_page():
    return render_template('ie_page.html')


# Information Extraction Result Page
@app.route('/information_extraction/result')
def ie_result_page():
    return render_template('ie_page.html', showResult=True)


# Sentiment Analysis Page
@app.route('/sentiment_analysis')
def sentiment_page():
    return render_template('sentiment_page.html')


# Result Page
@app.route('/result')
def result_page():
    return render_template('result_page.html', default=True)


# Result Page KNN
@app.route('/result/k_nearest_neighbor')
def result_page_knn():
    return render_template('result_page.html', knn=True)


# Result Page NB
@app.route('/result/naive_bayes')
def result_page_nb():
    return render_template('result_page.html', nb=True)


# Result Page DT
@app.route('/result/decision_tree')
def result_page_dt():
    return render_template('result_page.html', dt=True)


# Result Page RF
@app.route('/result/random_forest')
def result_page_rf():
    return render_template('result_page.html', rf=True)


# Result Page SVM
@app.route('/result/support_vector_machine')
def result_page_svm():
    return render_template('result_page.html', svm=True)


# Contact Page
@app.route('/contact')
def contact_page():
    return render_template('contact_page.html', th_language=True)


# Contact Page TH
@app.route("/contact/th", methods=['POST'])
def contact_th():
    return render_template('contact_page.html', th_language=True)


# Contact Page EN
@app.route("/contact/en", methods=['POST'])
def contact_en():
    return render_template('contact_page.html', th_language=False)


# '''''''''''''''''''''''''''''''''''
# '''''''''''' FUNCTION '''''''''''''
# '''''''''''''''''''''''''''''''''''
def df_html(df):
    table_html = """
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.22/css/jquery.dataTables.css"/>
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.22/js/jquery.dataTables.js"></script>
    <meta http-equiv="Content-type" content="text/html; charset=utf-8">
    %s<script type="text/javascript">$(document).ready(function(){$('table').DataTable({
        "pageLength": 10
    });});</script>
    """
    df_html = df.to_html(
        classes='table table-striped table-hover', justify='center', index=False)
    return table_html % df_html


# ======================== For Information Extraction ========================
def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    for token in tweet_tokens:
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub('RT', '', token)
        token = re.sub('(@[A-Za-z0-9_]+)', '', token)
        token = re.sub('(#[A-Za-z0-9_]+)', '', token)

        token = re.sub('“', '', token)
        token = re.sub('”', '', token)
        token = re.sub('’', '', token)
        token = re.sub('—', '', token)

        token = emoji_pattern.sub(r'', token)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_features(index, word, sent):
    prev_word = 'BOS'
    next_word = 'EOS'
    if len(sent) > index+1:
        next_word = sent[index+1]
    if index-1 > 0:
        prev_word = sent[index-1]
    val, tag = pos_tag([word])[0]
    prev_word, prev_tag = pos_tag([prev_word])[0]
    next_word, next_tag = pos_tag([next_word])[0]
    dic = {
        "word": val,
        "postag": tag,
        "nextword": next_word,
        "nextwordtag": next_tag,
        "previousword": prev_word,
        "previoustag": prev_tag,
    }
    return dic


def random_forest_predict(tokenized_sentence):
    new_list = []
    for i in range(len(tokenized_sentence)):
        new_list.append(get_features(
            i, tokenized_sentence[i], tokenized_sentence))

    pred = rf_model.predict(vec.transform(new_list))
    return pred


def extract_info(predicted_list, df_list, df):
    df_result = pd.DataFrame(
        columns=['Date', 'Cases', 'Deaths', 'Text'], dtype=int)

    for i in range(len(predicted_list)):
        result_list = [0] * 3
        for j in range(len(predicted_list[i])):

            # เมื่อพบ Tag 1 จะใส่ตัวเลขจำนวนผู้ติดเชื้อ และถ้ากรณีที่ไม่ใช่ตัวเลขจะให้ค่าเป็น 0
            if predicted_list[i][j] == 1:
                try:
                    result_list[0] = int(df_list[i][j].replace(',', ''))
                except:
                    result_list[0] = 0

            # เมื่อพบ Tag 2 จะใส่ตัวเลขจำนวนผู้เสียชีวิต และถ้ากรณีที่ไม่ใช่ตัวเลขจะให้ค่าเป็น 0
            elif predicted_list[i][j] == 2:
                try:
                    result_list[1] = int(df_list[i][j].replace(',', ''))
                except:
                    result_list[0] = 0

            # ใส่ข้อความดั้งเดิมลงไป
            result_list[2] = df['full_text'][i]
        # เพิ่มทีละ Row ไปเรื่อย ๆ จนครบ
        df_result.loc[i] = [df['created_at'][i]] + result_list
    return df_result


# ======================== For Sentiment Analysis ========================
def clean_text(sentence):
    stop_words = stopwords.words('english')
    tweet_tokens = TweetTokenizer().tokenize(str(sentence))

    final_token = []
    for token in tweet_tokens:
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub('RT', '', token)
        token = re.sub('(@[A-Za-z0-9_]+)', '', token)
        token = re.sub('(#[A-Za-z0-9_]+)', '', token)

        token = re.sub('“', '', token)
        token = re.sub('”', '', token)
        token = re.sub('’', '', token)
        token = re.sub('—', '', token)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            final_token.append(token.lower())

    return ' '.join(final_token)


def add_end_token(sentence):
    sent = sentence + ' <end>'
    return sent


def prepare_sequence_word(input_text):
    idxs = list()
    for word in input_text:
        if word in word2idx:
            idxs.append(word2idx[word])
        else:
            idxs.append(word2idx['UNK'])  # Use UNK tag for unknown word
    return idxs


def decode_sentiment(score):
    return 1 if score > 0.5 else 0


def bi_gru_predict(df_list):
    max_len_sentence = 100
    predicted_list = []

    for sent in df_list:
        predict = [prepare_sequence_word(sent)]
        predict = pad_sequences(maxlen=max_len_sentence, sequences=predict,
                                value=word2idx["UNK"], padding='post', truncating='post')

        # predict use model 1
        predicted = model.predict(predict)
        p = decode_sentiment(predicted)
        predicted_list.append(p)

    return predicted_list


def decode_result(label):
    decode = {0: "POSITIVE", 1: "NEGATIVE"}
    return decode[int(label)]


# '''''''''''''''''''''''''''''''''''
# '''''''''''' IE_PAGE ''''''''''''''
# '''''''''''''''''''''''''''''''''''
@app.route('/ie_retrieve_twitter_data', methods=['POST'])
def ie_retrieve_twitter_data():
    auth = tweepy.OAuthHandler(config.consumerKey, config.consumerSecret)
    auth.set_access_token(config.accessToken, config.accessTokenSecret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    country = request.form.get('country')
    session['country'] = country

    q = country + ' AND total AND COVID'
    df = pd.DataFrame(columns=['created_at', 'full_text'])
    for tweet in tweepy.Cursor(api.search,
                               q=q + ' -filter:retweets',
                               lang='en',
                               tweet_mode='extended').items(1000):

        created_at = str(tweet.created_at).split(' ')[0]
        try:
            full_text = str(tweet.retweeted_status.full_text)
        except:
            full_text = str(tweet.full_text)

        new_column = pd.Series([created_at, full_text], index=df.columns)
        df = df.append(new_column, ignore_index=True)

    return render_template('ie_page.html', df=df.to_json(), df_result=df_html(df), country=session['country'], readyToClean=True)


@app.route('/ie_data_cleaning', methods=['POST'])
def ie_data_cleaning():
    df = request.form.get('df')
    df = pd.read_json(df)

    # prepare to list
    df_column_list = list(df['created_at'].values)
    df_list = list(df['full_text'].values)
    # tokenize to sentence
    df_list = [TweetTokenizer().tokenize(str(sent).replace(',', ''))
               for sent in df_list]

    # cleaned token
    stop_words = stopwords.words('english')
    for i in range(len(df_list)):
        df_list[i] = remove_noise(df_list[i], stop_words)
    for i in range(len(df_list)):
        df_list[i].append('<end>')

    df = pd.DataFrame()
    df['created_at'] = df_column_list
    df['full_text'] = df_list

    return render_template('ie_page.html', cleaned_df=df.to_json(), show_cleaned_df=df_html(df), country=session['country'], readyToPredict=True)


@app.route('/ie_predict', methods=['POST'])
def ie_predict():
    df = request.form.get('cleaned_df')
    df = pd.read_json(df)

    df_list = list(df['full_text'].values)

    # predict
    predicted = [random_forest_predict(sent) for sent in df_list]

    # result
    df_result = extract_info(predicted, df_list, df)
    df_result['Date'] = df_result['Date'].apply(
        lambda x: str(x).split(' ')[0])
    df_result = df_result.loc[(df_result['Cases'] != 0)
                              & (df_result['Deaths'] != 0)]
    df_result = df_result.groupby(
        'Date', sort=True).mean().astype(int).reset_index()

    # plot line chart
    line_chart = pygal.Line(width=1250, height=600)
    line_chart.title = 'จำนวนคนติดเชื้อและเสียชีวิตทั้งหมดใน ' + \
        session['country'] + ' (เป็นจำนวนคน)'
    line_chart.x_labels = df_result['Date']
    line_chart.add('ติดเชื้อทั้งหมด', df_result['Cases'])
    line_chart.add('เสียชีวิตทั้งหมด', df_result['Deaths'])
    line_chart = line_chart.render_data_uri()

    df_result['Cases'] = df_result['Cases'].apply(
        lambda x: "{:,}".format(x))
    df_result['Deaths'] = df_result['Deaths'].apply(
        lambda x: "{:,}".format(x))

    return render_template('ie_page.html', line_chart=line_chart, predicted_df=df_html(df_result), country=session['country'])


# '''''''''''''''''''''''''''''''''''
# ''''''''' SENTIMENT_PAGE ''''''''''
# '''''''''''''''''''''''''''''''''''
@app.route('/sentiment_retrieve_twitter_data', methods=['POST'])
def sentiment_retrieve_twitter_data():
    auth = tweepy.OAuthHandler(config.consumerKey, config.consumerSecret)
    auth.set_access_token(config.accessToken, config.accessTokenSecret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    disease = request.form.get('disease')
    session['disease'] = disease

    q = disease + ' OR ' + '#' + disease
    df = pd.DataFrame(columns=['created_at', 'full_text'])
    for tweet in tweepy.Cursor(api.search,
                               q=q + ' -filter:retweets',
                               lang='en',
                               tweet_mode='extended').items(1000):

        created_at = str(tweet.created_at).split(' ')[0]
        try:
            full_text = str(tweet.retweeted_status.full_text)
        except:
            full_text = str(tweet.full_text)

        new_column = pd.Series([created_at, full_text], index=df.columns)
        df = df.append(new_column, ignore_index=True)

    return render_template('sentiment_page.html', df=df.to_json(), df_result=df_html(df), disease=session['disease'], sentimentReadyToClean=True)


@app.route('/sentiment_data_cleaning', methods=['POST'])
def sentiment_data_cleaning():
    df = request.form.get('df')
    df = pd.read_json(df)

    df['full_text'] = df['full_text'].apply(lambda x: clean_text(x))
    df['full_text'] = df['full_text'].apply(lambda x: add_end_token(x))

    return render_template('sentiment_page.html', cleaned_df=df.to_json(), show_cleaned_df=df_html(df), disease=session['disease'], sentimentReadyToPredict=True)


@app.route('/sentiment_predict', methods=['POST'])
def sentiment_predict():
    df = request.form.get('cleaned_df')
    df = pd.read_json(df)

    df_list = list(df['full_text'].values)

    # bi-gru predict
    predicted = bi_gru_predict(df_list)

    # result
    df['sentiment'] = [decode_result(x) for x in predicted]
    values = list(df['sentiment'].value_counts().keys())
    percentage = list(df['sentiment'].value_counts(normalize=True) * 100)

    # plot graph
    pie_chart = pygal.Pie(width=1200, height=600)
    pie_chart.title = 'อารมณ์ความรู้สึกของผู้คนต่อ ' + \
        session['disease'] + ' (เป็นเปอร์เซ็น %)'
    pie_chart.add('' + values[0], percentage[0])
    pie_chart.add('' + values[1], percentage[1])
    pie_chart = pie_chart.render_data_uri()

    return render_template('sentiment_page.html', pie_chart=pie_chart, predicted_df=df_html(df), disease=session['disease'])


if __name__ == '__main__':
    app.run(debug=True)
