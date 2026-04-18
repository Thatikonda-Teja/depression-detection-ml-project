from django.conf import settings
import pandas as pd
import numpy as np
import os
import re

# Use os.path.join for cross-platform compatibility
path = os.path.join(settings.MEDIA_ROOT, 'depression.csv')
path1 = os.path.join(settings.MEDIA_ROOT, 'no_depression.csv')

# --- Data Loading ---
try:
    clean = pd.read_csv(path)
    not_clean = pd.read_csv(path1)
    clean['class'] = 1
    not_clean['class'] = 0
    combined_data = pd.concat([clean, not_clean])
    new_data = combined_data[['tweet', 'class']]
except Exception as e:
    print(f"Error loading CSV files: {e}")
    print(f"Looking for files at: {path} and {path1}")
    new_data = None

# --- Text Cleaning ---
TEXT_CLEANING_RE = r"@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

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

# --- Preprocessing & Model Training (done once at module load) ---
x_train_vec = None
x_test_vec = None
y_train = None
y_test = None
tf = None

# Pre-trained model caches
_nb_model = None
_ec_model = None
_svc_model = None

if new_data is not None:
    try:
        new_data = new_data.dropna()
        new_data['tweet'] = new_data['tweet'].apply(clean_text)

        from nltk.stem import WordNetLemmatizer
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

        lematizer = WordNetLemmatizer()

        def lemmatizer_words(text):
            return " ".join([lematizer.lemmatize(word) for word in text.split()])

        new_data['tweet'] = new_data['tweet'].apply(lambda text: lemmatizer_words(text))

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(
            new_data['tweet'], new_data['class'], test_size=0.2, random_state=45
        )
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf = TfidfVectorizer()
        x_train_vec = tf.fit_transform(x_train)
        x_test_vec = tf.transform(x_test)

        # Pre-train all models once
        print("Training models at startup (one-time)...")

        from sklearn.naive_bayes import GaussianNB
        _nb_model = GaussianNB()
        _nb_model.fit(x_train_vec.toarray(), y_train)
        print("  GaussianNB trained.")

        from sklearn.ensemble import ExtraTreesClassifier
        _ec_model = ExtraTreesClassifier()
        _ec_model.fit(x_train_vec.toarray(), y_train)
        print("  ExtraTreesClassifier trained.")

        from sklearn.svm import SVC
        _svc_model = SVC()
        _svc_model.fit(x_train_vec.toarray(), y_train)
        print("  SVC trained.")

        print("All models ready.")

    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        x_train_vec = None
        x_test_vec = None
        y_train = None
        y_test = None
        tf = None


def GaussinNB():
    if x_train_vec is None or y_train is None or _nb_model is None:
        return 0, 0, 0, 0

    try:
        y_pred = _nb_model.predict(x_test_vec.toarray())
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision1 = precision_score(y_test, y_pred) * 100
        recall1 = recall_score(y_test, y_pred) * 100
        f1score1 = f1_score(y_test, y_pred) * 100
        print(f'NB - Accuracy: {accuracy}, Precision: {precision1}, Recall: {recall1}, F1: {f1score1}')
        return accuracy, precision1, recall1, f1score1
    except Exception as e:
        print(f"Error in GaussinNB: {e}")
        return 0, 0, 0, 0


def hybrid_model():
    if x_train_vec is None or y_train is None or _ec_model is None:
        return 0, 0, 0, 0

    try:
        y_pred = _ec_model.predict(x_test_vec.toarray())
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision1 = precision_score(y_test, y_pred) * 100
        recall1 = recall_score(y_test, y_pred) * 100
        f1score1 = f1_score(y_test, y_pred) * 100
        print(f'Hybrid - Accuracy: {accuracy}, Precision: {precision1}, Recall: {recall1}, F1: {f1score1}')
        return accuracy, precision1, recall1, f1score1
    except Exception as e:
        print(f"Error in hybrid_model: {e}")
        return 0, 0, 0, 0


def predict(posting):
    if _svc_model is None or tf is None:
        return 'error - model not loaded'

    try:
        input_text = [posting]
        input_data_features = tf.transform(input_text)
        prediction = _svc_model.predict(input_data_features.toarray())
        print(f"Prediction: {prediction}")
        if prediction[0] == 1:
            result = 'depression'
        else:
            result = 'no depression'
        return result
    except Exception as e:
        print(f"Error in predict: {e}")
        return 'error'