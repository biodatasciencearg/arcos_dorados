import json
import os
from flask import Flask,jsonify,request
from flask_cors import CORS
from predictor import my_sentiment_predictor

app = Flask(__name__)
CORS(app)
@app.route("/sentiment/",methods=['GET', 'POST'])
def return_pred():
  content = request.get_json(silent=True)
  print(content)
  data    = content["data"] 
  results,proba,label = my_sentiment_predictor.predict(data) 
  sentiment_dict = {
                'results': results,
                'proba' : proba,
                'label' : label
                }
  return jsonify(sentiment_dict)

@app.route("/",methods=['GET'])
def default():
  return "<h1> Welcome to sentiment predictor by E. Lopez <h1>"

if __name__ == "__main__":
    app.run() 
