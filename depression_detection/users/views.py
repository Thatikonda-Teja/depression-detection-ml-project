from django.shortcuts import render, HttpResponse
from django.contrib import messages
from sklearn.tree import DecisionTreeClassifier
from .forms import UserRegistrationForm
from .models import UserRegistrationModel


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def view_data(request):
    from django.conf import settings
    import pandas as pd
    import numpy as np
    path = settings.MEDIA_ROOT + '\\' + 'depression.csv'
    path1 = settings.MEDIA_ROOT + '\\' + 'no_depression.csv'
    clean = pd.read_csv(path)
    not_clean = pd.read_csv(path1)
    clean['class'] = 1
    not_clean['class'] = 0
    combined_data = pd.concat([clean,not_clean])
    combined_data.reset_index()
    combined_data = combined_data.to_html()
    return render(request,'users/view.html',{'data':combined_data})

def preprocess(request):
    from django.conf import settings
    import pandas as pd
    import numpy as np
    path = settings.MEDIA_ROOT + '\\' + 'depression.csv'
    path1 = settings.MEDIA_ROOT + '\\' + 'no_depression.csv'
    clean = pd.read_csv(path)
    not_clean = pd.read_csv(path1)
    clean['class'] = 1
    not_clean['class'] = 0
    combined_data = pd.concat([clean,not_clean])
    new_data = combined_data[['tweet','class']]

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

    new_data = new_data.dropna()
    new_data['tweet'] = new_data['tweet'].apply(clean_text)

    from nltk.stem import WordNetLemmatizer

    lematizer=WordNetLemmatizer()

    def lemmatizer_words(text):
        return " ".join([lematizer.lemmatize(word) for word in text.split()])

    new_data['tweet']=new_data['tweet'].apply(lambda text: lemmatizer_words(text)) 
    new_data = new_data.to_html()
    return render(request,'users/preprocess.html',{'preprocess':new_data})

def training(request):
    from .utility.resnet_50 import GaussinNB
    accuracy,precision1,recall1,f1score1 = GaussinNB()
    from .utility.resnet_50 import hybrid_model
    accuracy1,precision2,recall2,f1score2 = hybrid_model()
    
    return render(request,'users/ml.html',{'accuracy':accuracy,'precision1':precision1,'recall1':recall1,'f1score1':f1score1,'accuracy1':accuracy1,'precision2':precision2,'recall2':recall2,'f1score2':f1score2})

def user_input_prediction(request):
    if request.method=='POST':
        from .utility.resnet_50 import predict
        joninfo  = request.POST.get('joninfo')
        result = predict(joninfo)
        print(request)
        return render(request, 'users/testform.html', {'result': result})
    else:
        return render(request,'users/testform.html',{})
    

