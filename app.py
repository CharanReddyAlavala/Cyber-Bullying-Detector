import os
from flask import Flask, render_template, request
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer with error handling
try:
    model = tf.keras.models.load_model('lstm_model.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

try:
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    tokenizer = None
    print(f"Error loading tokenizer: {e}")

# Preprocessing function
def preprocess_comment(comment):
    comment = comment.lower()
    comment = re.sub(r'[^\w\s]', '', comment)
    return comment.strip()

@app.route('/', methods=['GET', 'POST'])
def detect_comment():
    prediction = None
    user_input = ""

    if request.method == 'POST':
        if 'detect' in request.form:
            user_input = request.form['comment']
            if model and tokenizer:
                processed_comment = preprocess_comment(user_input)
                sequences = tokenizer.texts_to_sequences([processed_comment])
                max_len = model.input_shape[1] if model.input_shape else 100
                padded_sequences = pad_sequences(sequences, maxlen=max_len)
                prediction_label = model.predict(padded_sequences)[0][0]
                prediction = "Cyberbullying" if prediction_label > 0.5 else "Non-Cyberbullying"
            else:
                prediction = "Error: Model or tokenizer not loaded."
        elif 'delete' in request.form:
            user_input = ""
            prediction = None

    return render_template('index.html', user_input=user_input, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
