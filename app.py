#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    def dataprocessing(text):
        # text = str(text)
        # print(text)
        
        text = text.lower()
        print(type(text))
        text = re.sub('<br />', '', text)
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text_tokens = word_tokenize(text)
        filtered_text = [w for w in text_tokens if not w in stop_words]
        return " ".join(filtered_text)

    def stemming(data):
        text = [stemmer.stem(word) for word in data]
        return data

    features = list(request.form.values())[0]
    # print(features[0])
    features = dataprocessing(features)
    stemmer = PorterStemmer()
    features = stemming(features)
    final_features = vectorizer.transform([features])
    prediction = model.predict(final_features)
    print(prediction[0])
    if prediction[0] ==0:
        output ="Negative"
    elif prediction[0] ==1:
        output ="Positive"
    else:
        output ="Neutral"

    
    return render_template('index.html', prediction_text='This Movie review is showing {} comments'.format(output))

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




