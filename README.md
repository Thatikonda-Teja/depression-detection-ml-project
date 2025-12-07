# Depression Detection Using NLP & Machine Learning

A Django-based web application that detects depression indicators from text/tweets using Natural Language Processing (NLP) and Machine Learning algorithms including Naive Bayes and Hybrid models.

## 📋 Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [ML Algorithms](#ml-algorithms)
- [Deployment](#deployment)

## ✨ Features

- **User Authentication**: Secure registration and login system
- **Text-Based Depression Detection**: Analyzes user-provided text for depression indicators
- **Multiple ML Models**: Gaussian Naive Bayes & ExtraTreesClassifier hybrid model
- **Data Visualization**: View and analyze training dataset
- **Text Preprocessing**: Tokenization, lemmatization, stopword removal
- **Admin Dashboard**: Manage users and monitor activity
- **Performance Metrics**: Accuracy, Precision, Recall, and F1-Score reporting
- **Responsive UI**: Bootstrap-based modern interface

## 🛠️ Tech Stack

- **Backend**: Django 4.1.3
- **NLP Libraries**: NLTK, Scikit-learn, WordCloud
- **ML Algorithms**: Naive Bayes, ExtraTreesClassifier
- **Text Processing**: TF-IDF Vectorization, Lemmatization, Regex
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Database**: SQLite3
- **Server**: Gunicorn
- **Deployment**: Render, Railway, Heroku

## 📦 Prerequisites

- Python 3.9+
- pip
- Virtual environment (recommended)
- Git

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/depression_detection.git
cd depression_detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Navigate to Django Project
```bash
cd depression_detection
```

### 5. Run Migrations
```bash
python manage.py migrate
```

### 6. Create Superuser (Admin)
```bash
python manage.py createsuperuser
```

### 7. Collect Static Files
```bash
python manage.py collectstatic --noinput
```

### 8. Run Development Server
```bash
python manage.py runserver
```

Visit `http://localhost:8000` in your browser.

## 💡 Usage

### User Features
1. **Register**: Create account with email, phone, and address
2. **Login**: Access dashboard with unique login credentials
3. **Enter Text**: Submit tweet or text content for analysis
4. **Get Prediction**: Receive depression detection results
5. **View Data**: Explore the training dataset (depression.csv & no_depression.csv)
6. **See Preprocessing**: View data cleaning and preprocessing steps

### Admin Features
1. **Access Admin Panel**: `http://localhost:8000/admin`
2. **View Registered Users**: Monitor all user registrations
3. **Manage Users**: Activate/deactivate accounts
4. **View Statistics**: Check model performance metrics

## 📁 Project Structure

```
depression_detection/
├── depression_detection/          # Main Django project
│   ├── settings.py               # Django configuration
│   ├── urls.py                   # URL routing
│   ├── views.py                  # Project views
│   ├── wsgi.py                   # WSGI configuration
│   └── asgi.py                   # ASGI configuration
├── users/                         # User app
│   ├── models.py                 # UserRegistrationModel
│   ├── views.py                  # User views & logic
│   ├── forms.py                  # User registration form
│   ├── utility/
│   │   └── resnet_50.py         # ML model & preprocessing
│   └── migrations/               # Database migrations
├── admins/                        # Admin app
│   ├── models.py                 # Admin models
│   ├── views.py                  # Admin views
│   └── migrations/               # Database migrations
├── templates/                     # HTML templates
│   ├── base.html                 # Base template
│   ├── index.html                # Home page
│   ├── UserLogin.html            # Login page
│   ├── UserRegistrations.html    # Registration page
│   ├── users/                    # User templates
│   │   ├── UserHomePage.html    # User dashboard
│   │   ├── testform.html        # Text input form
│   │   ├── view.html            # Data visualization
│   │   ├── preprocess.html      # Preprocessing display
│   │   ├── ml.html              # ML results
│   │   └── userbase.html        # User base template
│   └── admins/                   # Admin templates
│       ├── AdminHome.html
│       ├── AdminLogin.html
│       └── viewregisterusers.html
├── static/                        # Static files
│   ├── css/style.css
│   ├── js/main.js
│   ├── img/
│   └── vendor/
├── media/                         # Dataset
│   ├── depression.csv            # Depression text data
│   └── no_depression.csv         # Non-depression text data
├── requirements.txt              # Python dependencies
├── manage.py                      # Django CLI
├── Procfile                      # Deployment config
├── runtime.txt                   # Python version
└── README.md                     # Documentation
```

## 🧠 ML Models & Algorithms

### Data Preprocessing Pipeline
1. **Text Cleaning**: 
   - Convert to lowercase
   - Remove Twitter mentions (@user)
   - Remove URLs (http/https)
   - Remove special characters
   - Remove extra whitespace

2. **Stopword Removal**: 
   - Remove common English words (the, is, a, etc.)
   - Custom Twitter stopwords (rt, mkr, bc, etc.)

3. **Lemmatization**: 
   - Convert words to base form (running → run)
   - Uses NLTK WordNetLemmatizer

4. **Vectorization**: 
   - TF-IDF (Term Frequency-Inverse Document Frequency)
   - Converts text to numerical features (0-1 range)

### Model 1: Gaussian Naive Bayes
- **Type**: Probabilistic classifier
- **Algorithm**: Naive Bayes with Gaussian assumption
- **Training**: Converts TF-IDF vectors to dense arrays
- **Prediction**: Probability-based classification
- **Use Case**: Fast, efficient baseline model

### Model 2: Hybrid Model (ExtraTreesClassifier)
- **Type**: Ensemble learning (Extra Trees)
- **Algorithm**: Multiple decision trees with randomization
- **Features**: Handles non-linear relationships better
- **Prediction**: Combines multiple tree predictions
- **Use Case**: Improved accuracy over Naive Bayes

### Performance Metrics
- **Accuracy**: Overall correctness percentage
- **Precision**: True positives / All positive predictions
- **Recall**: True positives / All actual positives
- **F1-Score**: Harmonic mean of Precision & Recall

### Dataset
- **depression.csv**: Contains depressive tweets/text (Class=1)
- **no_depression.csv**: Contains non-depressive text (Class=0)
- **Train/Test Split**: 80% training, 20% testing
- **Random State**: 45 (for reproducibility)

## 🌐 Deployment

### Environment Variables Required
```
DEBUG=False
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=yourdomain.com
```

### Option 1: Render (Recommended)
1. Push to GitHub
2. Create account on render.com
3. Connect GitHub repository
4. Set environment variables
5. Deploy automatically

### Option 2: Railway
1. Push to GitHub
2. Visit railway.app
3. Create new project from GitHub
4. Auto-deploys on every push

### Option 3: Heroku
```bash
heroku login
heroku create app-name
git push heroku main
```

## 📊 API/URL Endpoints

| URL | Method | Purpose |
|-----|--------|---------|
| `/` | GET | Home page |
| `/register` | POST | User registration |
| `/login` | POST | User login |
| `/home` | GET | User dashboard |
| `/testform` | GET/POST | Text input form |
| `/predict` | POST | ML prediction |
| `/view` | GET | View dataset |
| `/preprocess` | GET | View preprocessing |
| `/admin` | GET | Admin panel |

## 🔒 Security Features

- CSRF protection
- SQL injection prevention via Django ORM
- Secure password handling
- Environment variable configuration
- Session-based authentication

## 🐛 Troubleshooting

### ImportError: No module named 'nltk'
```bash
pip install nltk
python -m nltk.downloader wordnet stopwords omw-1.4
```

### TF-IDF fitting error
Ensure depression.csv and no_depression.csv are in `media/` folder

### Static files not loading
```bash
cd depression_detection
python manage.py collectstatic --noinput
```

## 📝 Requirements

```
Django==4.1.3
python-decouple==3.8
Pillow==9.5.0
scikit-learn==1.3.0
numpy==1.24.0
pandas==2.0.0
nltk==3.8.1
wordcloud==1.9.2
gunicorn==21.2.0
whitenoise==6.5.0
```

## 👥 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open Pull Request

## 📄 License

MIT License - Free to use for educational and commercial purposes.

## 📧 Contact

For questions or improvements, please open an issue or contact the maintainers.

## 🙏 Acknowledgments

- Django Framework
- Scikit-learn documentation
- NLTK library
- Bootstrap framework
