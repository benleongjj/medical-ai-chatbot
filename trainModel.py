import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout 
from tensorflow.keras.optimizers.legacy import SGD

corpus = json.loads(open('chatbot_corpus.json').read())


lemmatizer = WordNetLemmatizer()

wordsInCorpus = []
tagsInCorpus = []
sentencesInCorpus = []
stop_letters = ['?', '!',',','.',"'"]


for intent in corpus['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        wordsInCorpus.extend(word_list) 
        sentencesInCorpus.append((word_list, intent['tag'])) 

        if intent['tag'] not in tagsInCorpus:
            tagsInCorpus.append(intent['tag']) 

wordsInCorpus = [lemmatizer.lemmatize(word.lower()) for word in wordsInCorpus if word not in stop_letters]
wordsInCorpus = sorted(set(wordsInCorpus))
tagsInCorpus = sorted(set(tagsInCorpus))


pickle.dump(wordsInCorpus, open('words.pkl','wb'))
pickle.dump(tagsInCorpus, open('classes.pkl','wb'))


trainingSet = []
tag_encoding = [0] *  len(tagsInCorpus)


for sentenceWTag in sentencesInCorpus:
    word_bag =[]
    
    word_patterns = sentenceWTag[0] 
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in wordsInCorpus:
        word_bag.append(1) if word in word_patterns else word_bag.append(0)
    
    output_row = list(tag_encoding)
    output_row[tagsInCorpus.index(sentenceWTag[1])] = 1 
    trainingSet.append([word_bag, output_row])

random.shuffle(trainingSet)
trainingSet = np.array(trainingSet, dtype=object)

train_x = list(trainingSet[:,0])
train_y = list(trainingSet[:,1])

model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('chatbotmodel.h5', hist)
print('Finished!')

