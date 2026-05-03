from flask import Flask, render_template, request, jsonify
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    L = []
    for i in text:
        if i.isalnum():
            L.append(i)

    text = L.copy()
    L.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            L.append(i)

    text = L.copy()
    L.clear()

    for i in text:
        L.append(ps.stem(i))

    return " ".join(L)

# Load model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# API route (FIXED)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get('message', '')

        transformed = transform_text(message)
        vector_input = tfidf.transform([transformed])
        prediction = model.predict(vector_input)[0]

        result = "Spam" if prediction == 1 else "Not Spam"

       

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

