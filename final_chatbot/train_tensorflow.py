import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('wordnet')

# Load intents data
intents = json.loads(open('intents2.json').read())

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []

# Define punctuation to ignore during tokenization
ignoreLetters = ['?','!','.',',']

# Process intents data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize patterns
        wordList = nltk.word_tokenize(pattern)
        # Add tokens to words list
        words.extend(wordList)
        # Add tokenized words and intent tag to documents list
        documents.append((wordList, intent['tag']))
        # Add intent tag to classes list if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]

# Remove duplicates and sort
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training data
training = []
outputEmpty = [0] * len(classes)

# Create training dataset
for document in documents:
    bag = []
    wordPatterens = document[0]
    wordPatterens = [lemmatizer.lemmatize(word.lower()) for word in wordPatterens]
    for word in words:
        bag.append(1) if word in wordPatterens else bag.append(0)
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle training data
random.shuffle(training)
training = np.array(training)

# Split data into features and labels
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

# Compile model
optimizer = tf.keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train model
history = model.fit(trainX, trainY, epochs=300, batch_size=25, verbose=1)

# Plot accuracy and loss
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Metrics')
plt.legend()
plt.show()

# Save model and training history
model.save('chat_model.h5')
with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

print('Training complete.')
