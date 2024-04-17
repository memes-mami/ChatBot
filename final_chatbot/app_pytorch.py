# Import necessary libraries
from flask import Flask, render_template, request, jsonify, redirect, url_for
import random
import json
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify
import random
import json

import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import re  # Import the re module


# Initialize Flask app
app = Flask(__name__)

# Load intents and model data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents2.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Alphius"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."

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
    resp = request.json['message']
    resp = get_response(resp)

    return jsonify({'response': resp})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
