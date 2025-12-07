from django.conf import settings
import pandas as pd
import numpy as np
import os

# Use os.path.join for cross-platform compatibility
path = os.path.join(settings.MEDIA_ROOT, 'depression.csv')
path1 = os.path.join(settings.MEDIA_ROOT, 'no_depression.csv')

try:
    clean = pd.read_csv(path)
    not_clean = pd.read_csv(path1)
    clean['class'] = 1
    not_clean['class'] = 0
    combined_data = pd.concat([clean,not_clean])
    new_data = combined_data[['tweet','class']]
except Exception as e:
    print(f"Error loading CSV files: {e}")
    print(f"Looking for files at: {path} and {path1}")
    new_data = None

import re
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

from wordcloud import STOPWORDS
STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 'im', 'll', 'y', 've', 'u', 'ur', 'don', 't', 's'])

def lower(text):
    return text.lower()

def remove_twitter(text):
    return re.sub(TEXT_CLEANING_RE, ' ', text)

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def clean_text(text):
    text = lower(text)
    text = remove_twitter(text)
    text = remove_stopwords(text)
    return text

if new_data is not None:
    try:
        new_data = new_data.dropna()
        new_data['tweet'] = new_data['tweet'].apply(clean_text)

        from nltk.stem import WordNetLemmatizer

        lematizer=WordNetLemmatizer()

        def lemmatizer_words(text):
            return " ".join([lematizer.lemmatize(word) for word in text.split()])

        new_data['tweet']=new_data['tweet'].apply(lambda text: lemmatizer_words(text))  

        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test = train_test_split(new_data['tweet'],new_data['class'],test_size=0.2,random_state=45)
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf = TfidfVectorizer()
        x_train_vec = tf.fit_transform(x_train)
        x_test_vec = tf.transform(x_test)
    except Exception as e:
        print(f"Error processing data: {e}")
        x_train_vec = None
        x_test_vec = None
        y_train = None
        y_test = None
        tf = None
else:
    x_train_vec = None
    x_test_vec = None
    y_train = None
    y_test = None
    tf = None

def GaussinNB():
    if x_train_vec is None or y_train is None:
        return 0, 0, 0, 0
    
    try:
        from sklearn.naive_bayes import GaussianNB
        nb = GaussianNB()
        nb.fit(x_train_vec.toarray(),y_train)
        y_pred = nb.predict(x_test_vec.toarray())
        from sklearn.metrics import accuracy_score,classification_report
        accuracy = accuracy_score(y_test, y_pred) * 100
        print('Accuracy:', accuracy)
        from sklearn.metrics import precision_score
        precision1 = precision_score(y_test, y_pred) * 100
        print('Precision Score:', precision1)
        from sklearn.metrics import recall_score
        recall1 = recall_score(y_test, y_pred) * 100
        print('recall_score:',recall1)
        from sklearn.metrics import f1_score
        f1score1 = f1_score(y_test, y_pred) * 100
        print('f1score:',f1score1)
        return accuracy,precision1,recall1,f1score1
    except Exception as e:
        print(f"Error in GaussinNB: {e}")
        return 0, 0, 0, 0

def hybrid_model():
    if x_train_vec is None or y_train is None:
        return 0, 0, 0, 0
    
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        EC = ExtraTreesClassifier()
        EC.fit(x_train_vec.toarray(),y_train)
        y_pred = EC.predict(x_test_vec.toarray())
        from sklearn.metrics import accuracy_score,classification_report
        accuracy = accuracy_score(y_test, y_pred) * 100
        print('Accuracy:', accuracy)
        from sklearn.metrics import precision_score
        precision1 = precision_score(y_test, y_pred) * 100
        print('Precision Score:', precision1)
        from sklearn.metrics import recall_score
        recall1 = recall_score(y_test, y_pred) * 100
        print('recall_score:',recall1)
        from sklearn.metrics import f1_score
        f1score1 = f1_score(y_test, y_pred) * 100
        print('f1score:',f1score1)
        return accuracy,precision1,recall1,f1score1
    except Exception as e:
        print(f"Error in hybrid_model: {e}")
        return 0, 0, 0, 0


    


def predict(posting):
    if x_train_vec is None or tf is None:
        return 'error'  # Return default if model not loaded
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC
        nb = SVC()
        nb.fit(x_train_vec.toarray(),y_train)

        input_text = [posting]
        input_data_features = tf.transform(input_text)
        # making prediction
        prediction = nb.predict(input_data_features.toarray())
        print(prediction)
        if (prediction[0] == 1):
            result = 'depression'
        else:
            result = 'no depression'
        return result
    except Exception as e:
        print(f"Error in predict: {e}")
        return 'error'