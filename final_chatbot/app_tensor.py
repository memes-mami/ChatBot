# Import necessary libraries
from flask import Flask, render_template, request, jsonify, redirect, url_for
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import re  # Import the re module

# Initialize Flask app
app = Flask(__name__)

# Load necessary resources for the chatbot (intents, model, etc.)
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents2.json').read())
model = tf.keras.models.load_model('chat_model.h5')
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# Define functions to clean up sentences and predict classes
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_classes(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in result]
    return return_list

def get_response(intents_list, all_intents):
    tag = intents_list[0]['intent']
    list_of_intents =  all_intents['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Define route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    warning_message = None
    if request.method == 'POST':
        email_format = '^[^\s@]+@[^\s@]+\.[^\s@]+$'
        email = request.form['email'].strip()
        if not re.match(email_format, email) or not email.endswith('@student.nitandhra.ac.in'):
            warning_message = "Invalid email format or domain. Please enter a valid *@student.nitandhra.ac.in email."
        else:
            return redirect(url_for('slideshow'))  # Redirect to basetemp2 page
    return render_template('index.html', warning_message=warning_message)

# Define route for basetemp2 page
@app.route('/basetemp2')
def slideshow():
    im = [
        'cv.png',
        'cloud.jpg',
        'tree.jpg',
        'aud.jpg',
        'ent.jpg',
        'mmm.jpg',
        'srk1.jpg'
    ]
    return render_template('basetemp2.html', images=im)

# Define route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Define route to process incoming messages
@app.route('/process_message', methods=['POST'])
def process_message():
    message = request.json['message']
    ints = predict_classes(message)
    response = get_response(ints, intents)
    print(response)
    return jsonify({'response': response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
