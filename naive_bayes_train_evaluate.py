from pyparsing import alphanums
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pickle

def read_data(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y

def train(X, Y, alpha):
    clf = MultinomialNB(alpha=alpha)
    clf.fit(X, Y)
    return clf

def get_acc(model, X, Y):
    return model.score(X, Y)

def save_model(model: MultinomialNB, filename):
    pickle.dump(model, open(filename, "wb"))
    

X, Y = read_data("train.txt.gz")
Xvalid, Yvalid = read_data("valid.txt.gz")

alpha_vals = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
best_valid = 0
for alpha in alpha_vals:
    model = train(X, Y, alpha)
    print("alpha: ", alpha)
    print("train acc: ", get_acc(model, X, Y))
    valid_acc = get_acc(model, Xvalid, Yvalid)
    print("valid acc: ", valid_acc)
    if valid_acc > best_valid:
        best_valid = valid_acc
        best_model = model
    print()
    save_model(model, "models_nb/MultinomialNB" + str(model.alpha) + ".sav")

save_model(best_model, "models_nb/BESTMultinomialNB" + str(best_model.alpha) + ".sav")
