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

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# NegativeFood model
class NegativeFood(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('review.id'), nullable=False)

# NegativeSanitary model
class NegativeSanitary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('review.id'), nullable=False)

# PositiveFood model
class PositiveFood(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('review.id'), nullable=False)

# PositiveSanitary model
class PositiveSanitary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('review.id'), nullable=False)

# Review model
class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(20), nullable=False)
    
    positive_food = db.relationship('PositiveFood', backref='review', lazy=True)
    negative_food = db.relationship('NegativeFood', backref='review', lazy=True)
    positive_sanitary = db.relationship('PositiveSanitary', backref='review', lazy=True)
    negative_sanitary = db.relationship('NegativeSanitary', backref='review', lazy=True)

#Users model
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
    return render_template('index.html')

# Register route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            is_admin = request.form.get('admin') == 'yes'

            # Define Regex patterns
            password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'
            email_pattern = r'^[\w\.-]+@[\w\.-]+(\.[\w]+)+$'

            if is_admin:
                admin_exists = User.query.filter_by(is_admin=True).first()
                if admin_exists:
                    flash('An admin user already exists. You cannot sign up as an admin.', 'error')
                    return redirect(url_for('login'))
            
            # Check if the username and email are unique
            existing_user = User.query.filter_by(username=username).first()
            existing_email = User.query.filter_by(email=email).first()
            if username == '' or email == '' or password == '':
                flash('Please fill in your details', 'error')
            elif existing_user:
                flash('Username already exists. Please choose a different username.', 'error')
            elif existing_email:
                flash('Email address already registered. Please use a different email.', 'error')
            
            # Check password and email validity using Regex
            elif not re.match(password_pattern, password):
                flash('Password must be at least 8 characters long and contain at least one letter and one digit.', 'error')
            elif not re.match(email_pattern, email):
                flash('Invalid email format. Please enter a valid email address.', 'error')
            else:
                print('Else statement ran successfully')
                hashed_password = generate_password_hash(password, method='sha256')
                new_user = User(username=username, email=email, password=hashed_password, is_admin=is_admin)
                db.session.add(new_user)
                db.session.commit()
                flash('Account created successfully. You can now log in.', 'success')
                return redirect(url_for('login'))
        except Exception as e:
            flash('Something went wrong', 'error')
            print(str(e))
            db.session.rollback()
        finally:
            db.session.close()
    
    return render_template('login.html')

# login route
@login_manager.user_loader
def load_user(user_id):
    user = db.session.get(User, int(user_id))
    return user

@app.route('/login', methods=['GET', 'POST'])
def login():
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
data = pd.read_csv('Sentiment_Analysis.csv')
reviews = data['word'].tolist()
labels = data['category'].tolist()

# Preprocess the text data
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)
preprocessed_reviews = [preprocess_text(review) for review in reviews]

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(preprocessed_reviews)
labels = labels
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.4)

classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)
            
@app.route('/submit_review', methods=['GET', 'POST'])
@login_required
def submit_review():
    if request.method == 'POST':
        try:
            text = request.form['Review_comment']

            user_review = preprocess_text(text)
            user_review_vector = tfidf_vectorizer.transform([user_review])
            predicted_category = classifier.predict(user_review_vector)
            review = Review(user_id=current_user.id, text=text, category=predicted_category[0])

            if text == '':
                flash('You cannot submit an empty review', 'error')
            elif len(text) < 3:
                flash('The text must be at least three characters long', 'error')
            else:
                db.session.add(review)
                db.session.commit()
                flash('Review submitted successfully', 'success')
                return redirect(url_for('submit_review'))
        except Exception as e:
            flash('Something went wrong')
            print(str(e))
            db.session.rollback()
        finally:
            db.session.close()
    loaded_reviews = Review.query.all()
    return render_template('reviews.html', loaded_reviews=loaded_reviews)

# admin route
@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    categories = ['Positive Food', 'Negative Food', 'Positive Sanitary', 'Negative Sanitary']
    review_counts = defaultdict(int)
    
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

    p = create_bokeh_plot(review_counts)
    script, div = components(p)
    all_reviews = Review.query.all()
    return render_template('admin_dashboard.html', script=script, div=div, reviews = all_reviews)

# User Profile route
@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

# Update User Profile route
@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    if request.method == 'POST':
        current_user.username = request.form.get('username')
        current_user.email = request.form.get('email')

        db.session.commit()
        flash('Profile updated successfully', 'success')

    return redirect(url_for('profile'))

@app.route('/delete_profile', methods=['GET', 'POST'])
@login_required
def delete_profile():
    if request.method == 'POST':
        db.session.delete(current_user)
        db.session.commit()
        logout_user()

        flash('Your profile has been deleted successfully', 'success')
        return redirect(url_for('index'))
    return render_template('confirm_delete.html')

# logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('index'))

# app context
with app.app_context():
    db.create_all()

# main method
if __name__ == '__main__':
    app.run(debug=True)