import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import joblib

nltk.download('stopwords')

f = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv', delimiter = ',')
t = pd.read_csv('../input/fake-and-real-news-dataset/True.csv', delimiter = ',')

f['temp']= 0
t['temp']= 1

datas = pd.DataFrame()
datas = t.append(f)

column = ['date','subject']
datas = datas.drop(columns=column)

input_arr=np.array(datas['title'])

corpus = []
for i in range(0, 40000):
    newArr = re.sub('[^a-zA-Z]', ' ', input_arr[i])
    newArr = newArr.lower()
    newArr = newArr.split()
    ps = PorterStemmer()
    newArr = [ps.stem(word) for word in newArr if not word in set(stopwords.words('english'))]
    newArr = ' '.join(newArr)
    corpus.append(newArr)

countv = CountVectorizer(max_features = 5000)
X = countv.fit_transform(corpus).toarray()
y = datas.iloc[0:40000, 2].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

joblib.dump(classifier, 'logistic_regression_model.pkl')
