# These are import statements that import various libraries and modules that are used in the code.
from flask import render_template, request, redirect, url_for, flash, Flask
from flask_login import login_user, login_required, logout_user, current_user, LoginManager, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from collections import defaultdict
from bokeh.embed import components
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from config import Config
from utils import create_bokeh_plot
import re

# Initialize Flask instance
# `app = Flask(__name__)` creates an instance of the Flask class, which represents the Flask
# application. The `__name__` argument is a special Python variable that gets the name of the current
# module. This is necessary for Flask to know where to look for resources such as templates and static
# files.
app = Flask(__name__)

# Configure Flask instance with configuration settings
# `app.config.from_object(Config)` is used to load configuration settings from a Python object. In
# this case, it is loading the configuration settings from the `Config` class defined in the `config`
# module. The `Config` class contains various configuration variables that are used to configure the
# Flask application, such as the secret key, database URI, and other settings. By calling
# `app.config.from_object(Config)`, the Flask application is configured with the settings defined in
# the `Config` class.
app.config.from_object(Config)

# `db = SQLAlchemy(app)` creates an instance of the SQLAlchemy class and binds it to the Flask
# application `app`. SQLAlchemy is an Object-Relational Mapping (ORM) library that provides a set of
# tools for working with databases in Python. By binding the SQLAlchemy instance to the Flask
# application, we can use SQLAlchemy to interact with the database within our Flask application.
db = SQLAlchemy(app)

# Initialize login manager
# `login_manager = LoginManager()` creates an instance of the `LoginManager` class.
login_manager = LoginManager()
# `login_manager.login_view = 'login'` is setting the login view for the Flask-Login extension.
login_manager.login_view = 'login'
# `login_manager.init_app(app)` is initializing the Flask-Login extension with the Flask application
# `app`.
login_manager.init_app(app)

# NegativeFood model
# The class "NegativeFood" represents a model for negative food reviews in a database, with
# attributes for the ID and the associated review ID.
class NegativeFood(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('review.id'), nullable=False)

# NegativeSanitary model
# The class "NegativeSanitary" represents a model for negative sanitary reviews in a database, with
# attributes for the ID and the associated review ID.
class NegativeSanitary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('review.id'), nullable=False)

# PositiveFood model
# The class "PositiveFood" represents a model for positive food reviews in a database, with
# attributes for the ID and the associated review ID.
class PositiveFood(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('review.id'), nullable=False)

# PositiveSanitary model
# The class "PositiveSanitary" represents a model for positive sanitary reviews in a database, with
# attributes for the ID and the associated review ID.
class PositiveSanitary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('review.id'), nullable=False)

# Review model
# The Review class represents a review object with attributes such as id, user_id, text, and category,
# and has relationships with the PositiveFood, NegativeFood, PositiveSanitary, and NegativeSanitary
# classes.
class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(20), nullable=False)
    
    # relationships with food and sanitary tables
    # The below code is defining relationships between the `Review` model and other models
    # (`PositiveFood`, `NegativeFood`, `PositiveSanitary`, `NegativeSanitary`) in a database. These
    # relationships are defined using the `db.relationship` function and specify the backref and lazy
    # loading options.
    positive_food = db.relationship('PositiveFood', backref='review', lazy=True)
    negative_food = db.relationship('NegativeFood', backref='review', lazy=True)
    positive_sanitary = db.relationship('PositiveSanitary', backref='review', lazy=True)
    negative_sanitary = db.relationship('NegativeSanitary', backref='review', lazy=True)

#Users model
# The below class represents a User model with attributes such as id, username, email, password, and
# is_admin.
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

#----------------------------routes---------------------------------

# Index/Home route
@app.route('/')
@app.route('/home')
def index():
    """
    The function `index` is a route handler that renders the `index.html` template when the user visits
    the `/` or `/home` URL.
    :return: the rendered template 'index.html'.
    """
    return render_template('index.html')

# Register route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """
    The `signup` function is a route in a Flask application that handles the signup process, including
    form validation and creating a new user in the database.
    :return: the rendered template 'login.html' if the request method is 'GET'. If the request method is
    'POST', it will handle the form submission and redirect the user to the 'login' route.
    """
    if request.method == 'POST':
        try:
            # The below code is retrieving user input from a form and assigning it to variables. It is
            # getting the values of 'username', 'email', 'password' from the form and assigning them
            # to the respective variables. It is also checking if the value of 'admin' in the form is
            # 'yes' and assigning the boolean value to the variable 'is_admin'.
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            is_admin = request.form.get('admin') == 'yes'

            # Define Regex patterns
            # The line of code below is defining a regular expression pattern for validating passwords.
            # The pattern requires the password to have at least one digit and one letter, and be
            # at least 8 characters long.
            password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

            # The line of code below is defining a regular expression pattern for validating email
            # addresses. The pattern is checking for the following conditions:
            # - The email address should start with one or more word characters, dots, or hyphens.
            # - After the initial characters, there should be an "@" symbol.
            # - After the "@" symbol, there should be one or more word characters, dots, or
            # hyphens.
            # - Finally, there should be one or more occurrences of a dot followed by one or more
            # word characters.
            email_pattern = r'^[\w\.-]+@[\w\.-]+(\.[\w]+)+$'
            
            # Check if the username and email are unique
            # The below first two lines of code are querying the database to check if there is an existing 
            # user with the given username and email. It uses the `User.query.filter_by()` method to 
            # filter the User table by the username and email columns, and then uses the `first()` method 
            # to retrieve the first matching user from the query results.
            existing_user = User.query.filter_by(username=username).first()
            existing_email = User.query.filter_by(email=email).first()
            if username == '' or email == '' or password == '':
                flash('Please fill in your details', 'error')
            elif existing_user:
                flash('Username already exists. Please choose a different username.', 'error')
            elif existing_email:
                flash('Email address already registered. Please use a different email.', 'error')
            
            # Check password and email validity using Regex
            # The below code is checking if the variable `password` matches a specific pattern
            # defined by the regular expression `password_pattern`. If the password does not match
            # the pattern, the code will execute the block of code inside the `if` statement.
            elif not re.match(password_pattern, password):
                flash('Password must be at least 8 characters long and contain at least one letter and one digit.', 'error')
            
            # The below code is checking if the variable `email` does not match the regular
            # expression pattern `email_pattern`.
            elif not re.match(email_pattern, email):
                flash('Invalid email format. Please enter a valid email address.', 'error')
            else:
                print('Else statement ran successfully')
                # Create a new user
                # The below code is creating a new user in a database. It takes in a username, email,
                # password, and is_admin flag as input. It then generates a hashed password using the
                # SHA256 algorithm. Finally, it creates a new user object with the provided
                # information and adds it to the database.
                hashed_password = generate_password_hash(password, method='sha256')
                new_user = User(username=username, email=email, password=hashed_password, is_admin=is_admin)
                db.session.add(new_user)
                db.session.commit()
                flash('Account created successfully. You can now log in.', 'success')
                return redirect(url_for('login'))
        
        # The below code is handling an exception. It catches any exception that occurs and
        # assigns it to the variable `e`. It then displays a flash message with the text "Something
        # went wrong" and the category "error". The exception message is also printed to the console.
        # Finally, it rolls back any changes made to the database session.
        except Exception as e:
            flash('Something went wrong', 'error')
            print(str(e))
            db.session.rollback()
        
        # The below code is using the `finally` block to ensure that the `db.session.close()` method
        # is always called, regardless of whether an exception is raised or not. This is commonly used
        # in database operations to ensure that the database connection is properly closed after the
        # operation is completed.
        finally:
            db.session.close()
    
    return render_template('login.html')

# login route

# Define the user_loader callback function
@login_manager.user_loader
def load_user(user_id):
    """
    The function `load_user` is a decorator that loads a user from the database based on the user ID.
    
    :param user_id: The user_id parameter is the unique identifier for a user. It is used to load the
    user object from the database
    :return: The user object with the specified user_id is being returned.
    """
    user = db.session.get(User, int(user_id))
    return user

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    This is a login function that handles both GET and POST requests, checks if the user exists and if
    the password is correct, and logs in the user if successful.
    :return: the rendered template 'login.html' if the request method is 'GET'. If the request method is
    'POST', it will perform the login logic and redirect the user to different pages based on their role
    (admin or regular user).
    """
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
            
            # Check if the user exists
            user = User.query.filter_by(username=username).first()

            if username == '' or password == '':
                flash('Please fill in your details')
            elif user and check_password_hash(user.password, password):
                login_user(user, remember=True)
                if user.is_admin:
                    # Redirect the admin to the admin dashboard
                    return redirect(url_for('admin_dashboard'))
                flash('Logged in successfully.', 'success')
                return redirect(url_for('submit_review'))
            flash('Login failed. Please check your username and password.')
        except Exception as e:
            flash('Something went wrong. Please try again')
            print(str(e))
            db.session.rollback()
        finally:
            db.session.close()
    
    return render_template('login.html')

# review route
# Load and preprocess the dataset
# The below code is reading a CSV file named 'Sentiment_Analysis.csv' and storing its contents in a
# variable called 'data'.
data = pd.read_csv('Sentiment_Analysis.csv')

# Extract the text reviews and labels from the DataFrame
# The below code is converting the 'word' column of a DataFrame called 'data' into a list called
# 'reviews', and converting the 'category' column of the same DataFrame into a list called 'labels'.
reviews = data['word'].tolist()
labels = data['category'].tolist()

# Preprocess the text data
def preprocess_text(text):
    """
    The function preprocesses text by tokenizing it, removing stop words, and stemming the remaining
    words.
    
    :param text: The `text` parameter is a string that represents the input text that needs to be
    preprocessed
    :return: The function preprocess_text returns a string that is the result of joining the
    preprocessed tokens together.
    """

    # The Natural Language Toolkit (nltk) library tokenize a given text. 
    # Tokenization is the process of breaking down a text into individual words or tokens.
    # The code is using the `word_tokenize()` function from the nltk library to perform this
    # tokenization. The resulting tokens will be stored in the `tokens` variable.
    tokens = nltk.word_tokenize(text)

    # The line of code below uses the `stopwords` module from the `nltk` library and create a set of
    # stop words in English. Stop words are commonly used words (such as "the", "is", "and", etc.)
    # that are often removed from text during natural language processing tasks like text
    # classification or information retrieval.
    stop_words = set(stopwords.words('english'))

    # The line of code below is filtering out stop words from a list of tokens. It creates a new list called
    # "tokens" by iterating over each word in the original list and checking if the lowercase version
    # of the word is not in the stop_words list. If the condition is true, the word is added to the
    # new list.
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # The line of code below uses the PorterStemmer class from the nltk library and create an
    # instance of it called "stemmer".
    stemmer = PorterStemmer()

    # The below code is using a stemmer to perform stemming on a list of tokens. Stemming is the
    # process of reducing words to their base or root form (plural to singular). In this code, 
    # the stemmer is applied to each word in the list of tokens, and the resulting stemmed words 
    # are stored in a new list called "tokens" and joins them together with spaces in between.
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# The line of code below is creating a list called `preprocessed_reviews` by applying the function
# `preprocess_text` to each element in the `reviews` list.
preprocessed_reviews = [preprocess_text(review) for review in reviews]

# Feature extraction using TF-IDF
# The below code is performing TF-IDF vectorization on a set of preprocessed reviews. It uses the
# TfidfVectorizer class from the scikit-learn library to convert the text data into a numerical
# representation. The fit_transform() method is used to fit the vectorizer to the data and transform
# the reviews into a matrix of TF-IDF features. The resulting matrix, X, contains the TF-IDF values
# for each word in each review.
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(preprocessed_reviews)

# Assign labels to your dataset
labels = labels

# Split the dataset into training and testing sets
# The line of code below is splitting the dataset `X` and its corresponding labels `labels` into training and
# testing sets. The training set is stored in `X_train` and `y_train`, while the testing set is stored
# in `X_test` and `y_test`. The `test_size` parameter is set to 0.4, which means that 40% of the data
# will be used for testing and 60% will be used for training.
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.4)

# Train a classifier
# The below code is creating a support vector machine (SVM) classifier with a radial basis function
# (RBF) kernel. It then fits (trains) the classifier to the training data (X_train and y_train).
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)
            
@app.route('/submit_review', methods=['GET', 'POST'])
@login_required
def submit_review():
    """
    The `submit_review` function allows users to submit a review. It preprocesses the
    review text, extracts features, predicts the category of the review, and stores the review in the
    database.
    :return: the rendered template 'reviews.html' with the loaded_reviews (all reviews) variable passed as a
    parameter.
    """
    if request.method == 'POST':
        try:
            text = request.form['Review_comment']

             # Preprocess the user review
            user_review = preprocess_text(text)
            
            # Extract features for the user review
            user_review_vector = tfidf_vectorizer.transform([user_review])
            
            # Predict the category of the user's review
            # The line of code below is using a classifier to predict the category of a user review based on a
            # vector representation of the review.
            predicted_category = classifier.predict(user_review_vector)

            # The line of code below is creating a new `Review` object with the following attributes:
            # `user_id`, `text`, and `category`. The `user_id` is set to the `id` of the current user,
            # the `text` is set to the value of the `text` variable, and the `category` is set to the
            # first element of the `predicted_category` list.
            review = Review(user_id=current_user.id, text=text, category=predicted_category[0])

            if text == '':
                flash('You cannot submit an empty review', 'error')
            elif len(text) < 3:
                flash('The text must be at least three characters long', 'error')
            else:
                db.session.add(review)
                db.session.commit()
                flash('Review submitted successfully', 'success')

                #load reviews to the screen
                return redirect(url_for('submit_review'))
        except Exception as e:
            flash('Something went wrong')
            print(str(e))
            db.session.rollback()
        finally:
            db.session.close()
    
    # The line of code below is querying all the reviews from the database and storing them in the variable
    # `loaded_reviews`.
    loaded_reviews = Review.query.all()
    return render_template('reviews.html', loaded_reviews=loaded_reviews)

# admin route
@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    """
    The `admin_dashboard` function counts the number of reviews in each category and creates a Bokeh
    plot to display the results on the admin dashboard page.
    :return: the rendered template 'admin_dashboard.html' along with the Bokeh plot script and div
    components.
    """
    # Count the number of reviews in each category
    categories = ['Positive Food', 'Negative Food', 'Positive Sanitary', 'Negative Sanitary']
    review_counts = defaultdict(int)
    
    # The below code is iterating through a list of categories. For each category, it checks if it
    # matches a specific condition and then performs a count query on a Review table based on that
    # condition. The count query checks for specific keywords in the text column of the Review table.
    # The count result is then assigned to the corresponding category in a dictionary called
    # review_counts. If a category does not match any of the specified conditions, the count is set to
    # 0 for that category.
    for category in categories:
        if category == 'Positive Food':
            count = Review.query.filter((Review.category == 'Positive') & Review.text.ilike('%food%')).count()
        elif category == 'Negative Food':
            count = Review.query.filter((Review.category == 'Negative') & Review.text.ilike('%food%')).count()
        elif category == 'Positive Sanitary':
            count = Review.query.filter((Review.category == 'Positive') & (Review.text.ilike('%sanitary%') | Review.text.ilike('%sanitation%'))).count()
        elif category == 'Negative Sanitary':
            count = Review.query.filter((Review.category == 'Negative') & (Review.text.ilike('%sanitary%') | Review.text.ilike('%sanitation%'))).count()
        else:
            count = 0
        
        review_counts[category] = count
    
    # Create a Bokeh plot
    # The below code is creating a Bokeh plot using the `create_bokeh_plot` function and passing in the
    # `review_counts` data. It then generates the JavaScript and HTML code needed to embed the plot in
    # a webpage using the `components` function, and assigns the resulting script and div to the
    # variables `script` and `div` respectively.
    p = create_bokeh_plot(review_counts)
    script, div = components(p)
    
    return render_template('admin_dashboard.html', script=script, div=div)

# User Profile route
@app.route('/profile')
@login_required
def profile():
    """
    The `profile` function is a route handler for the user profile page. It is decorated with
    `@login_required` to ensure that only authenticated users can access their profiles.
    :return: the rendered template 'profile.html' with the current user's information.
    """
    return render_template('profile.html', user=current_user)

# Update User Profile route
@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    """
    The `update_profile` function is a route handler for updating the user's profile information.
    It is decorated with `@login_required` to ensure that only authenticated users can update their profiles.
    :return: a redirect to the 'profile' route after updating the user's profile.
    """
    if request.method == 'POST':
        # Retrieve and update user information
        current_user.username = request.form.get('username')
        current_user.email = request.form.get('email')

        # Commit the changes to the database
        db.session.commit()
        flash('Profile updated successfully', 'success')

    return redirect(url_for('profile'))

@app.route('/delete_profile', methods=['GET', 'POST'])
@login_required
def delete_profile():
    if request.method == 'POST':
        # Delete the user's account
        db.session.delete(current_user)
        db.session.commit()

        # Log the user out
        logout_user()

        # Redirect to a page indicating successful deletion
        flash('Your profile has been deleted successfully', 'success')
        return redirect(url_for('index'))

    # If the request method is GET, render the confirmation page
    return render_template('confirm_delete.html')

# logout route
@app.route('/logout')
@login_required
def logout():
    """
    The logout() function logs out the user, flashes a success message, and redirects them to the index
    page.
    :return: a redirect to the 'index' route.
    """
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('index'))

# app context
# The below code is using a context manager to create an application context for the current app. This
# allows the code within the context to access and interact with the application's resources and
# configurations. It then create all the tables defined in the database schema using the SQLAlchemy
# library in a Flask application context.
with app.app_context():
    db.create_all()

# main method
# The below code is checking if the current module is being run as the main module. If it is, it calls
# the `run` function of the `app` object with the `debug` parameter set to `True`.
if __name__ == '__main__':
    app.run(debug=True)