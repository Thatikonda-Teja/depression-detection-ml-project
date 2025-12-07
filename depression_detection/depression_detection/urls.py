"""depression_detection URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from depression_detection import views as mainView
from admins import views as admins
from users import views as usr


urlpatterns = [
    path('admin/', admin.site.urls),
    path("", mainView.index, name="index"),
    path("AdminLogin/", mainView.AdminLogin, name="AdminLogin"),
    path("UserLogin/", mainView.UserLogin, name="UserLogin"),
    path("UserRegister/", mainView.UserRegister, name="UserRegister"),

    # Adminviews
    path("AdminLoginCheck/", admins.AdminLoginCheck, name="AdminLoginCheck"),
    path("AdminHome/", admins.AdminHome, name="AdminHome"),
    path('RegisterUsersView/', admins.RegisterUsersView, name='RegisterUsersView'),
    path('ActivaUsers/', admins.ActivaUsers, name='ActivaUsers'),
    path('DeleteUsers/', admins.DeleteUsers, name='DeleteUsers'),

    # User Views
    path("UserRegisterActions/", usr.UserRegisterActions, name="UserRegisterActions"),
    path("UserLoginCheck/", usr.UserLoginCheck, name="UserLoginCheck"),
    path("view_data/", usr.view_data, name="view_data"),
    path("preprocess/", usr.preprocess, name="preprocess"),

    path("UserHome/", usr.UserHome, name="UserHome"),
    path("training/", usr.training, name="training"),
    path("user_input_prediction/", usr.user_input_prediction, name="user_input_prediction"),

    # path("PreProcess/", usr.PreProcess, name="PreProcess"),
    # path("View_Data/", usr.View_Data, name="View_Data"),
    # path("ML/", usr.ML, name="ML"),
    # path("MLResult/", usr.MLResult, name="MLResult"),
    # path("user_classification_results/", usr.user_classification_results, name="user_classification_results"),

]

# Serve static and media files in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
