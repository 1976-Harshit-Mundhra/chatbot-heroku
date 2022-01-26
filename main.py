# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
from tensorflow import keras
import numpy as np
import json 
from flask import Flask, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

path = 'new_datav4.json'

with open(path) as file : 
  data = json.load(file)
  
  def chat(abc):
    # load trained model
    chat_model = keras.models.load_model('bestv2.h5')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('lbl_encoder.pickle', 'rb') as enc:
        onehot_encoded = pickle.load(enc)

    # parameters
    max_len = 10
        
    inp = abc
    if inp.lower() == "quit":
        return "exit"
    
    result = chat_model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
    tag = onehot_encoded.inverse_transform([np.argmax(result)])
        

    for i in data['intents']:
        if i['tag'] == tag:
            print("ChatBot:", np.random.choice(i['responses']))
            return np.random.choice(i['responses'])    

#chat("what is trf")            

app = Flask(__name__)
api = Api(app)
CORS(app)

'''@app.route('/<string:name>/')
def hello(name):
    name = name.replace("_"," ")
    data = {'chatbot':chat(name)}
    return jsonify(data)'''

class chatbot(Resource):
    def get(self, name):
        name = name.replace("_"," ")
        return jsonify({'chatbot': chat(name)})
    
class status (Resource):
    def get(self):
        try:
            return {'data': 'Api is Running'}
        except:
            return {'data': 'An Error Occurred during fetching Api'}

api.add_resource(status, '/')
api.add_resource(chatbot, '/chatbot/<string:name>/')
   
if __name__ == '__main__':
    app.run(debug=True)           