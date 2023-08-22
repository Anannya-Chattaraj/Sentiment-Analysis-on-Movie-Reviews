from flask import Flask, render_template,request
from model import test_model
import pickle

app = Flask(__name__)
cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():

    if 'review_text' in request.form:
        if request.method=='POST':
             message = request.form['review_text']
             sentiment = test_model(message)

    return render_template("index.html", sentiment=sentiment)

if __name__ == '__main__':
  # Run the Flask app
  app.run(debug=True)     
        
    


