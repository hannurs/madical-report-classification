import glob
from operator import index
import os
import numpy as np
from porter import stem
from build_vocabulary import remove_punctuation

def get_classes():
    classes = dict()
    # for path in glob.glob("medical-reports" + "/**/*.txt", recursive=True):
    #     filename = os.path.basename(path)
    #     classname = filename[5:-4]
    #     classes[classname] = classes.get(classname, 0) + 1

    for path in glob.glob("medical-reports/train/" + "*.txt", recursive=False):
        filename = os.path.basename(path)
        classname = filename[5:-4]
        classes[classname] = classes.get(classname, 0) + 1

    classes_list = list(classes.keys())
    return classes_list
    # print(classes_list)

def load_vocabulary(filename):
    f = open(filename)
    words = f.read().split()
    f.close()
    voc = {}
    index = 0
    for word in words:
        voc[word] = index
        index += 1
    return voc

def extract_features(filename, voc):
    f = open(filename, encoding="utf-8")
    text = f.read().lower()
    f.close()
    text = remove_punctuation(text)
    bow = np.zeros(len(voc))
    for word in text.split():
        word = stem(word)
        if word in voc:
            index = voc[word]
            bow[index] += 1
    return bow

def meanvar_normalization(Xtrain, Xvalid, Xtest):
    mu = Xtrain.mean(0)
    std = Xtrain.std(0)
    Xtrain = (Xtrain - mu) / std
    Xvalid = (Xvalid - mu) / std
    Xtest = (Xtest - mu) / std
    return Xtrain, Xvalid, Xtest

classes_list = get_classes()
voc = load_vocabulary("vocabulary.txt")

features = []
labels = []
for f in os.listdir("medical-reports/train"):
    path = "medical-reports/train/" + f
    bow = extract_features(path, voc)
    features.append(bow)
    classname = f[5:-4]
    labels.append(classes_list.index(classname))
    # print(classname, ": ", classes_list.index(classname))



features_validation = []
labels_validation = []
for f in os.listdir("medical-reports/validation"):
    path = "medical-reports/validation/" + f
    bow = extract_features(path, voc)
    features_validation.append(bow)
    classname = f[5:-4]
    labels_validation.append(classes_list.index(classname))
    # print(classname, ": ", classes_list.index(classname))

features_test = []
labels_test = []
for f in os.listdir("medical-reports/test"):
    path = "medical-reports/test/" + f
    bow = extract_features(path, voc)
    features_test.append(bow)
    classname = f[5:-4]
    labels_test.append(classes_list.index(classname))
    # print(classname, ": ", classes_list.index(classname))

X = np.stack(features)
X_validation = np.stack(features_validation)
X_test = np.stack(features_test)
# X, X_validation, X_test = meanvar_normalization(X, X_validation, X_test)

Y = np.array(labels)
Y_validation = np.array(labels_validation)
Y_test = np.array(labels_test)

data = np.concatenate([X, Y[:, None]], 1)
data_validation = np.concatenate([X_validation, Y_validation[:, None]], 1)
data_test = np.concatenate([X_test, Y_test[:, None]], 1)

np.savetxt("train.txt.gz", data)
np.savetxt("valid.txt.gz", data_validation)
np.savetxt("test.txt.gz", data_test)

