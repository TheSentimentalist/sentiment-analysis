# sentiment-analysis

This project aims to analyze the emotion on the user reviews for the Come hotel (Not a real hotel).

# processes

1. Take the input from the user.
2. Apply the analytic model on the input.
3. Determines the emotion on the input (Positive or Negative)

# Analysis tools
1. nltk

# Algorithms
1. Support Vector Machines classifier

# Libraries
See <a href="requirements.txt">requirements.txt</a> file

# NOTE
Make sure to add config.py file to the root of the application.
- In the config.py file:
- class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///hotel_reviews.db' or any name you like other than hotel_reviews
    SQLALCHEMY_TRACK_MODIFICATIONS = False (or True if you want)
    SECRET_KEY = 'Add your secret key (just type anything)'