import random
import json
import numpy as np
import pickle
import nltk 
from nltk.stem import WordNetLemmatizer 
from tensorflow.keras.models import load_model

corpus = json.loads(open('chatbot_corpus.json').read())

wordsInCorpus = pickle.load(open('words.pkl','rb'))
tagsInCorpus = pickle.load(open('classes.pkl','rb'))

model = load_model('chatbotmodel.h5')


lemmatizer = WordNetLemmatizer()

def sentence_processing(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def convertToBOW(sentence):
    sentence_words = sentence_processing(sentence)
    bag = [0] * len(wordsInCorpus)
    for w in sentence_words:
        for i, word in enumerate(wordsInCorpus):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_tag(sentence):
    bow = convertToBOW(sentence)
    res = model.predict(np.array([bow]))[0]
    errThreshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > errThreshold]
    results.sort(key=lambda x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':tagsInCorpus[r[0]], 'probability':str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Hello, you may start typing!")

while True:
    message = input("User: ").lower()
    ints = predict_tag(message)
    res = get_response(ints, corpus)
    print("Chatbot: " + res)

