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


def get_synonyms(word):
    ps = PorterStemmer()
    synonyms=[]
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(ps.stem(str(l.name())))
    return synonyms



def generate_input():
    input = []
    files = ["malaria.txt", "dengue.txt", "typhoid.txt","viralfever.txt"]
    count = []
    for x in files:
        filepath = x
        count1 = 0
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                count1 = count1 + 1
                input.append(vectorize(line))
                line = fp.readline()
        count.append(count1)
        fp.close()
    return np.asarray(input), count

def decision_tree(X_train, X_test, Y_train, Y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    #print(model)

    # make predictions
    expected = Y_test
    predicted = model.predict(X_test)

    # summarize the fit of the model
    print("\nSVM Classifier\n")
    #print(metrics.classification_report(expected, predicted))
    print("Confusion Matrix: ")
    print(metrics.confusion_matrix(expected, predicted))
    print("Accuracy: ")
    print(accuracy_score(Y_test, predicted))
    filename = 'decision_tree.sav'
    pickle.dump(model, open(filename, 'wb'))

def svm_classifier(X_train, X_test, Y_train, Y_test):
    model = svm.SVC(kernel='rbf', gamma='scale')  # Linear Kernel
    model.fit(X_train, Y_train)
    #print(model)

    # make predictions
    expected = Y_test
    predicted = model.predict(X_test)

    # summarize the fit of the model
    print("\nDecision Tree Classifier\n")
    #print(metrics.classification_report(expected, predicted))
    print("Confusion Matrix: ")
    print(metrics.confusion_matrix(expected, predicted))
    print("Accuracy: ")
    print(accuracy_score(Y_test, predicted))
    filename = 'svm.sav'
    pickle.dump(model, open(filename, 'wb'))

def random_forest(X_train, X_test, Y_train, Y_test):
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
    rf.fit(X_train, Y_train)

    # make predictions\n
    Y_expect_RF = Y_test
    Y_predict_RF = rf.predict(X_test)

    # summarize the fit of the model
    print("\nRandom Forest\n")
    #print(metrics.classification_report(Y_expect_RF, Y_predict_RF))
    print("Confusion Matrix: ")
    print(metrics.confusion_matrix(Y_expect_RF, Y_predict_RF))
    print("Accuracy: ")
    print(accuracy_score(Y_expect_RF, Y_predict_RF))
    filename = 'random_forest.sav'
    pickle.dump(rf, open(filename, 'wb'))

def KMeans_Classifier(X_train, X_test, Y_train, Y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)
    Y_expect_RF = Y_test
    Y_predict_RF = knn.predict(X_test)
    print("\nK Means Classification\n")
    # print(metrics.classification_report(Y_expect_RF, Y_predict_RF))
    print("Confusion Matrix: ")
    print(metrics.confusion_matrix(Y_expect_RF, Y_predict_RF))
    print("Accuracy: ")
    print(accuracy_score(Y_expect_RF, Y_predict_RF))
    filename = 'Kmeans.sav'
    pickle.dump(knn, open(filename, 'wb'))

def Naive_Bayes(X_train, X_test, Y_train, Y_test):
    nb = MultinomialNB()
    nb.fit(X_train, Y_train)
    Y_expect_RF = Y_test
    Y_predict_RF = nb.predict(X_test)
    print("\nNaive Bayes\n")
    # print(metrics.classification_report(Y_expect_RF, Y_predict_RF))
    print("Confusion Matrix: ")
    print(metrics.confusion_matrix(Y_expect_RF, Y_predict_RF))
    print("Accuracy: ")
    print(accuracy_score(Y_expect_RF, Y_predict_RF))
    filename = 'naive_bayes.sav'
    pickle.dump(nb, open(filename, 'wb'))

def Logistic_Regression(X_train, X_test, Y_train, Y_test):
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    lr.fit(X_train, Y_train)
    Y_expect_RF = Y_test
    Y_predict_RF = lr.predict(X_test)
    print("\nLogistic Regression\n")
    # print(metrics.classification_report(Y_expect_RF, Y_predict_RF))
    print("Confusion Matrix: ")
    print(metrics.confusion_matrix(Y_expect_RF, Y_predict_RF))
    print("Accuracy: ")
    print(accuracy_score(Y_expect_RF, Y_predict_RF))
    filename = 'regression.sav'
    pickle.dump(lr, open(filename, 'wb'))


class AutoEncoder:
    def __init__(self, np_input, encoding_dim=8):
        self.encoding_dim = encoding_dim
        self.x = np_input

    def _encoder(self):
        inputs = Input(shape=(self.x[0].shape))
        encoded = Dense(16, activation='relu')(inputs)
        encoded = Dense(10, activation='relu')(encoded)
        #encoded = Dense(8, activation='relu')(encoded)
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded = Dense(16, activation= 'relu')(inputs)
        decoded = Dense(20, activation= 'relu') (decoded)
        #decoded = Dense(16, activation='relu')(decoded)
        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()

        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model
        return model

    def fit(self, batch_size=100, epochs=300):
        self.model.compile(optimizer='sgd', loss='mse')
        log_dir = './log2/'
        tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(self.x, self.x,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[tbCallBack])

    def save(self):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
        else:
            self.encoder.save(r'./weights/encoder_weights2.h5')
            self.decoder.save(r'./weights/decoder_weights2.h5')
            self.model.save(r'./weights/ae_weights2.h5')

    def encode_input(self, np_input):
        #inputs = np.array(np_input)
        x = self.encoder.predict(np_input)
        #y = self.decoder.predict(x)
        return x
        # print('Input: {}'.format(inputs))
        # print('Encoded: {}'.format(x))
        # print('Decoded: {}'.format(y))


if __name__ == '__main__':
    np_input,count = generate_input()
    ae = AutoEncoder(np_input, encoding_dim=10)
    ae.encoder_decoder()
    ae.fit(batch_size=20, epochs=300)
    ae.save()
    encoded_input= ae.encode_input(np_input).tolist()
    # encoder = load_model(r'./weights/encoder_weights2.h5')
    # encoded_input = encoder.predict(np_input).tolist()
    Y =  [4 if x<count[0] else 2 if x<count[0]+count[1] else 3 if x<count[0]+count[1]+count[2] else 5 for x in range(len(encoded_input))]
    X_train, X_test, Y_train, Y_test = train_test_split(encoded_input, Y, test_size=0.1, random_state= 109)
    decision_tree(X_train, X_test, Y_train, Y_test)
    svm_classifier(X_train, X_test, Y_train, Y_test)
    random_forest(X_train, X_test, Y_train, Y_test)
    KMeans_Classifier(X_train, X_test, Y_train, Y_test)
    Logistic_Regression(X_train, X_test, Y_train, Y_test)
    Naive_Bayes(X_train, X_test, Y_train, Y_test)
    string = input("\nEnter your symptoms\n")
    vector= vectorize(string)
    #print(vector)
    arr = np.empty((0, 20), int)
    arr = np.append(arr, np.array([vector]), axis=0)
    # encoder = load_model(r'./weights/encoder_weights2.h5')
    # input_encoded = encoder.predict(arr).tolist()
    input_encoded = ae.encode_input(arr)
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

