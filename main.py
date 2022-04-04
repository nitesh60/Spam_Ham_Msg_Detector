from flask import Flask, render_template, request
import re
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/submit", methods = ["POST"])
def get_data():
    if request.method == "POST":
        sms = request.form['txt1']

        # Data Cleaning
        sms = re.sub('[^a-zA-Z]', ' ', sms) 
        sms = sms.lower()
        sms = sms.split()

        # Stemming and Generating Corpus
        user_corpus = []
        ps = PorterStemmer()
        sms = [ps.stem(word) for word in sms if not word in stopwords.words('english')]
        sms = ' '.join(sms)
        user_corpus.append(sms)

        # Applying Count Vectorizer to Corpus
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        x = tfidf.transform(user_corpus)

        # Loading ML Model
        nb_model = pickle.load(open('nb_model.pkl','rb'))
        result = nb_model.predict(x)[0]
        res = "Spam" if result == 1 else "Ham"
        return render_template("Home.html", result = res)
        #return redirect("/", result = res)

    return render_template("Home.html")

if __name__ == "__main__":
    app.run(debug = True)