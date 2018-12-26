import keras
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
import numpy as np
from tensorflow import set_random_seed
import os
from nltk.tokenize import word_tokenize, sent_tokenize, MWETokenizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from sklearn.externals import joblib
import pickle
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize, MWETokenizer
from nltk.stem import PorterStemmer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_synonyms(word):
    ps = PorterStemmer()
    synonyms=[]
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(ps.stem(str(l.name())))
    return synonyms

def vectorize(patient):

    tokenizer = MWETokenizer([("bleeding", "gum"), ("chest", "pain"), ("abdominal", "pain"), ("muscle", "pain"),("joint", "pain"),("eye", "pain"), ("nerve", "pain"), ("ligament", "pain"), ("tendon", "pain"),
        ("bleeding", "nose")])

    dict = ["headach", "vomit", "nausea", "bleeding_gum", "itch", "rash", "fever", "diarrhea","discomfort","chest_pain", "abdominal_pain", "fatigu", "muscle_pain", "chill", "eye_pain", "joint_pain", "nerve_pain","ligament_pain", "tendon_pain", "bleeding_nos"]
    dict2 = ["headache", "vomit", "nausea", "bleeding_gum", "itch", "rash", "fever", "diarrhea", "discomfort",
             "chest_pain", "abdominal_pain", "fatigue", "muscle_pain", "chill", "eye_pain", "joint_pain", "nerve_pain",
             "ligament_pain", "tendon_pain", "bleeding_nose"]
    synonyms_dict=[get_synonyms(dict2[x]) for x in range(len(dict2))]
    tokens= tokenizer.tokenize(word_tokenize(patient))
    ps = PorterStemmer()
    modified_tokens=[ps.stem(word) for word in tokens]
    #print(modified_tokens)

    token_set=[]
    arra=[0 for x in range(len(dict))]

    for word in modified_tokens:
        for x in range(len(dict)):
            if word==dict[x]:
                token_set.append(word)
                arra[x]=1

            else:
                for x in range(len(dict)):
                    if word in synonyms_dict[x]:
                        token_set.append(word)
                        arra[x]=1
    return arra



def predict_disease(string):
    vector= vectorize(string)
    arr = np.empty((0, 20), int)
    arr = np.append(arr, np.array([vector]), axis=0)
    all_zeroes=np.all(arr==0)
    if all_zeroes==1:
        print("No fever disease")
    else:
        encoder = load_model(r'./weights/encoder_weights2.h5')
        input_encoded = encoder.predict(arr).tolist()
        loaded_model = joblib.load('decision_tree.sav')
        result = loaded_model.predict(input_encoded)
        result= ["Malaria" if result==[2] else "Dengue" if result==[3] else "Typhoid" if result==[4] else "Viral Fever"]
        print("Decision Tree: " + str(result[0]))
        loaded_model = joblib.load('svm.sav')
        result = loaded_model.predict(input_encoded)
        result = ["Malaria" if result == [2] else "Dengue" if result == [3] else "Typhoid" if result==[4] else "Viral Fever"]
        print("SVM: " + str(result[0]))
        loaded_model = joblib.load('Kmeans.sav')
        result = loaded_model.predict(input_encoded)
        result = ["Malaria" if result == [2] else "Dengue" if result == [3] else "Typhoid" if result==[4] else "Viral Fever"]
        print("Kmeans: " + str(result[0]))
        loaded_model = joblib.load('naive_bayes.sav')
        result = loaded_model.predict(input_encoded)
        result = ["Malaria" if result == [2] else "Dengue" if result == [3] else "Typhoid" if result==[4] else "Viral Fever"]
        print("Naive Bayes: " + str(result[0]))
        loaded_model = joblib.load('regression.sav')
        result = loaded_model.predict(input_encoded)
        result = ["Malaria" if result == [2] else "Dengue" if result == [3] else "Typhoid" if result==[4] else "Viral Fever"]
        print("regression: " + str(result[0]))
        loaded_model = joblib.load('random_forest.sav')
        result = loaded_model.predict(input_encoded)
        result = ["Malaria" if result == [2] else "Dengue" if result == [3] else "Typhoid" if result==[4] else "Viral Fever"]
        print("Random Forest: " + str(result[0]))



if __name__ == '__main__':
    complain = input("\nEnter your symptoms\n")
    predict_disease(complain)
    next_complain=input("\nCheck for next patient?")
    while(next_complain.lower()=='y'):
        complain = input("\nEnter your symptoms\n")
        predict_disease(complain)
        next_complain=input("\nCheck for next patient?(y/n)")

    