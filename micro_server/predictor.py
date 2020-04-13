import pickle
import pandas as pd
import numpy as np
import eli5
from eli5.lime import TextExplainer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
class my_sentiment_predictor():
    def __init__(self):
        pass

    def predict(data):
        with open('predictor.pkl', 'rb') as handle:
            text_clf,le = pickle.load(handle)
        te = TextExplainer(random_state=42)
        te.fit(data, text_clf.predict_proba)
        clases = list(set(le.classes_))
        results = te.show_prediction(target_names=clases).data.replace('\n',' ')
        proba   = "{0:.3f}".format((text_clf.predict_proba([data])[0][text_clf.predict([data])[0]])*100)
        label   = le.classes_[text_clf.predict([data])[0]]
        return results,proba,label
