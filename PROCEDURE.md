# Introduction

The SARS-Cov-2 known as Covid-19 has spread around the world and its also designation as a worldwide pandemic by the World Health Organization in March 2020. As the number of such cases increases, the availability of information regarding these diseases is important in order to help experts in taking proper actions and predict the pattern of the disease itself. Information extraction and sentiment analysis has been broadly acknowledged as one of the first stages in deep language understanding and its importance has been well recognized in the natural language processing community.

This study has objective in developing text classification system which classified the textual information on the social media platforms like Twitter. We propose two approaches for two tasks, the first approach is we apply classification method algorithm to information extraction task, we use K-NN, Naïve Bayes, Decision Tree, Random Forest and Support Vector Machine. Evaluate each algorithm, conclude a result of F1 score and accuracy score, and compare them together to find which one has better result. Our second approach we use is deep learning method for sentiment analysis task, we apply one of the most well known neural network method call Bi-directional GRU, we use two different word embedding and evaluate them.

# Related Work

Several studies and research have been conducted regarding text categorization and relatively simple NLP approach to analyzing the tweet content. Experiment on n-gram models for classifying related or not related to the flu by Alex Lamb, Michael J. Paul and Mark Dredze [[1]](#1). Bayu Yudha Pratama and Riyanarto Sarno propose a Naive Bayes,  KNN and SVM for classifying the personality trait [[2]](#2). Farhad Nooralahzadeh shows the experiment for linking the mentions of named entities also known as Entity Linking based on Knowledge Bases [[3]](#3). Fatimah Wulandini and Anto Satriyo Nugroho evaluated the performance of Text Classification using SVM and 3 other conventional methods [[4]](#4). Giridhar Kumaran and James Allan said addressing named entities preferentially is useful only in certain situations [[5]](#5). Hila Becker, Mor Naaman and Luis Gravano explain about a short messages posted on social media can typically reflect events as they happen [[6]](#6). John Pestian [[7]](#7) Mihai Dusmanu, Elena Cabrio and Serena Villata [[8]](#8) Min Song [[9]](#9) Yi Zhang and Bing Liu [[12]](#12) Yin Aphinyanaphongs [[13]](#13) show how to pre process the data by using simple NLP i.e part of speech tagging, twitter-specific syntax, tokenization, lemmatization. Vincent Van Asch introduce the implementaion and evaluation of precision, recall and f1-score both micro-averaged and macro-averaged [[10]](#10). Wei-jie Guan and the others provide the information about Covid-19 how patients often presented without fever and many did not have abnormal radiologic findings [[11]](#11).

# Procedure

- Dataset and Pre Processing <br>

In information extraction we have collected almost 600,000 tweets by using python library name Tweepy, since 15 Feb to 29 Feb 2020 with keyword “#COVID19”. We pre processed it by cleaning and tokenization text using NLTK library, we select the tweets that has the particular word like “cases”, “infected”, “dead”, “death” and “died” occur in the sentence, almost 600,000 tweets have been lowered to around 150,000 tweets by doing these process. In sentiment analysis the dataset that we use is from Kaggle (“Sentiment140 dataset with 1.6 million tweets”, KazAnova, 2017), the dataset is not relevant to Covid-19 but it has 1.6 million tweets and the polarity of the tweet (Negative, Positive), we also pre processed it by cleaning text and tokenization using NLTK library.

- Information Extraction <br>

The task consists in classifying a tweets as containing report information of coronavirus. Our interest focuses in particular on tweets pattern like “total 42 cases” or “500 total deaths” of their sources. We are going to use the technique that is call labeling, we labeling 700 tweets from 150,000 tweets in Feb manually. And using 5 algorithms of classification as follows K-NN, Naïve Bays, Decision Tree, Random Forest and Support Vector Machine to classify and evaluate the result.

We need to find the amount of people who affected to this pandemic, then we must annotated the numbers that occur in text. This would allow us to understand and make it easier to train the algorithms. These numbers are annotated as “1” if it follows by “total cases”, or we annotated as “2” if it follows by “total deaths”, if it not fits the above conditions then we annotated as “0” (see example (a) and (b) below).

(a) Text: [iran, reports, 3, new, cases, bringing, total, confirmed, cases, 5, 2, total, deaths] <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tag: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]
  
(b) Text: [total, deaths, 75, total, cases, 100] <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tag: [0, 0, 2, 0, 0, 1]

After manually labeling, we then prepare it to algorithms by transforming text to Python dictionary by having 6 features as the following.

<p align="center">
  <img src="../main/readme-img/table1.PNG" />
</p>

Where “word” is the word itself,  “postag” is the part of speech of the word, “nextword” is the next word after the word itself, “nextwordtag” is the next part of speech of the next word, “previousword” is the previous word before the word itself and “previoustag” is the previous part of speech before the word itself. The part of speech tag we use is NLTK postag.

We split the dataset to 70:30 training and testing data and transform Python dictionary to vectors by using DictVectorizer from Scikit-learn library. Typically, when the feature values are numeric we can use One-hot Encoder to complete binary one-hot encoding. But our feature values are strings, we then propose a DictVectorizer instead. This transformer  turns list of feature-value mappings to vectors, this will do a binary one-hot coding also know as one-of-K. Table 2 shows the result precision, recall and F1-score of all algorithms.

<p align="center">
  <img src="../main/readme-img/table2.PNG" />
</p>

It is likely give us a pretty good result. But most of the flaw of the algorithms happened when testing in the real-time data (e.g. Twitter), it not recognized the information sources. In other word, when the source is coming from Twitter happen to be an opinion or an argument this approach will not work anymore. However, in order to draw more interesting conclusions in this task, we would need to increase the size of the dataset.

- Sentiment Analysis <br>

We propose a Bi-directional GRU sentiment analysis classification model. We transform every word into index by iterating them, we split it to 70:30 training and testing data before give it to our model. Image below shows the construction process of the Bi-directional GRU sentiment analysis classification model.

<p align="center">
  <img src="../main/readme-img/figure1.png" width="50%" height="50%"/>
</p>

Where the embedding layer is the layer for pre-trained word embedding, dropout layer is for prevent overfitting by set input to 0 each step during training with particular rate, bi-directional gru layer is the model bi-directional with gated recurrent unit, this model separate sequence to 2 direction, one from left to right and the other right to left and concatenating together, fully-connected layer is basic neural network layer which complies the data by previous layer to form the final output, sigmoid layer is consider as the output from the fully-connected layer, the output use sigmoid function activation which return the result value in the range 0 to 1.

<p align="center">
  <img src="../main/readme-img/table345.PNG"/>
</p>

it shows us that the English word embedding has better accuracy than Covid-19 word embedding, because our dataset we use to train is not related to Covid-19.

# Discussion and Future work

This study investigated information extraction and sentiment analysis on Twitter data. The main goal is to study and sharing the work of every method that we are used, we propose a very few approaches on each two main tasks and evaluate them. These tasks are particularly relevant when applied to social media data and the Covid-19 global pandemic.

The main issue of information extraction on Twitter is the dataset, we are labeling the dataset by manually unlike the sentiment analysis task we use a ready-to-use dataset. Thus, the dataset on information extraction is limited, which give the algorithms not comprehensive to a real-time text, bringing us to limited result and accuracy. Although, even sentiment analysis has a ready-to-use dataset it still has the problem. The main problem is Kaggle dataset that we downloaded is not relevant to Covid-19, it will give us low accuracy to classify the sentiment in real-time text that related to Covid-19. Both tasks need to improve and perfect in the next experiment.

In future work. We will focus on extending and increasing the datasets of information extraction by augmentation method, and exploring more on sentiment analysis dataset in order to have more reliability in real-time use.

# References

<a id="1">[1]</a> 
Alex Lamb. (2013). Separating Fact from Fear: Tracking Flu Infections on Twitter. Johns Hopkins University. <br>
<a id="2">[2]</a> 
Bayu Yudha Pratama. (2018). Personality Classification Based on Twitter Text using Naive 	Bayes, KNN and SVM. University of Indonesia. <br>
<a id="3">[3]</a> 
Farhad Nooralahzadeh. (2017). Adapting Semantic Spreading Activation to Entity Linking in Text. INRIA Sophia Antipolis. <br>
<a id="4">[4]</a> 
Fatimah Wulandini. (2009). Text Classification Using Support Vector Machine for 	Webmining Based Spatio Temporal Analysis of the Spread of Tropical Diseases. 	Swiss German University. <br>
<a id="5">[5]</a>
Giridhar Kumaran. (2004). Text Classification and Named Entities for New Event Detection. University of Massachusetts Amherst. <br>
<a id="6">[6]</a>
Hila Becker. (2011). Beyond Trending Topics: Real-World Event Identification on Twitter. Columbia University. <br>
<a id="7">[7]</a> 
John Pestian. (2010). Suicide Note Classification using Natural Language Processing: A Content Analysis. Cincinnati Children’s Hospital Medical Center. <br>
<a id="8">[8]</a> 
Mihai Dusmanu. (2017). Argument Mining on Twitter: Arguments, Facts and Sources. 	University Cote d'Azur. <br>
<a id="9">[9]</a> 
Min Song. (2015). Entity and relation extraction for public knowledge discovery. Yonsei University. <br>
<a id="10">[10]</a> 
Vincent Van Asch. (2013). Macro- and micro-averaged evaluation measures [[BASIC 	DRAFT]]. University of Antwerp. <br>
<a id="11">[11]</a> 
Wei-jie Guan. (2019). Clinical Characteristics of Coronavirus Disease 2019 in China.	Hospital of Guangzhou Medical University. <br>
<a id="12">[12]</a> 
Yi Zhang. (2007). Semantic Text Classification of Emergent Disease Reports. University of Illinois. <br>
<a id="13">[13]</a> 
Yin Aphinyanophongs. (2016). Text Classification for Automatic Detection of E-Cigarette Use and Use For Smoking Cessation From Twitter: A Feasibility Pilot. NYU Langone Medical Center.