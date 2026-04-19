#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (needed for text processing)
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Navigate to Django project directory
cd depression_detection

# Collect static files
python manage.py collectstatic --noinput

# Run database migrations
python manage.py migrate
