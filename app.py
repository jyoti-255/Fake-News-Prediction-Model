# Import libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import nltk

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Load the data with caching to improve performance
@st.cache_data
def load_data():
    data = pd.read_csv('train.csv')
    data = data.fillna(' ')
    data['content'] = data['author'] + " " + data['title']
    return data

data = load_data()

# Check for null values
print(data.isnull().sum())

# Preprocessing
def stemming(content):
    ps = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stop_words = set(stopwords.words('english'))
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    return " ".join(stemmed_content)

# Apply stemming with caching to avoid repeated computations
@st.cache_data
def preprocess_data(data):
    data['content'] = data['content'].apply(stemming)
    return data

data = preprocess_data(data)

# Separate the dataset
x = data['content'].values
y = data['label'].values

# Vectorization
vector = TfidfVectorizer()
x = vector.fit_transform(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# Model training with caching to avoid retraining on every run
@st.cache_resource
def train_model(_x_train, _y_train):
    model = LogisticRegression()
    model.fit(_x_train, _y_train)
    return model

model = train_model(x_train, y_train)

# Streamlit app
st.title("Fake News Detector")
input_text = st.text_input("Enter news article")

# Prediction function
def predict(input_text):
    input_text_transformed = vector.transform([input_text])
    prediction = model.predict(input_text_transformed)
    return prediction[0]

if input_text:
    prediction_result = predict(input_text)
    if prediction_result == 1:
        st.write("The News is Fake")
    else:
        st.write("The News is Real")





