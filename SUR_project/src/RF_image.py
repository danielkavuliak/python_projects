import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier


def eval_model(directory, clf):
    file_name = []
    X = []
    
    # ziskame zoznam suborov z priecinku, aby sme mohli nacitavat obrazky
    images = os.listdir(directory)
    for img in images:
        if img.split('.')[-1] == 'png':
            X.append(ImageOps.grayscale(Image.open(directory + '/' + img)))
            file_name.append(img.split('.')[0])
            
    # obrazky predspracovavame tak ako vo funkcii preprocess_dataset
    converter = transforms.PILToTensor()
    for i in range(len(X)):
        tmp = converter(X[i]).double().numpy()
        X[i] = tmp[0].reshape(tmp[0].shape[0] * tmp[0].shape[1])

    X = np.asarray(X)
    X -= X.mean(axis=0)
    X -= X.mean(axis=1).reshape(X.shape[0], -1)
    
    # nakoniec, vypocitame pravdepodobnosti prislusnosti obrazkov do tried
    # do textoveho suboru zapisujeme udaje, ktore boli uvedene v zadani projektu
    probabilities = clf.predict_proba(X)
    with open('RF_image.txt', 'w') as f:
        is_detected = None
        
        for i in range(len(file_name)):
            if probabilities[i][1] >= 0.5:
                is_detected = 1
            else:
                is_detected = 0
                
            f.write(file_name[i] + ' ' + str(probabilities[i][1]) + ' ' + str(is_detected) + '\n')


def preprocess_dataset(X_train, X_test):
    converter = transforms.PILToTensor()

    # vsetky obrazky prevedieme z PIL obrazku do tensoru, nasledne do numpy pola a napokon ich pretransformujeme do vektora
    for i in range(len(X_train)):
        tmp = converter(X_train[i]).double().numpy()
        X_train[i] = tmp[0].reshape(tmp[0].shape[0] * tmp[0].shape[1])

    for i in range(len(X_test)):
        tmp = converter(X_test[i]).double().numpy()
        X_test[i] = tmp[0].reshape(tmp[0].shape[0] * tmp[0].shape[1])

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    
    # od kazdeho obrazku odcitame stredne hodnoty priznakov, aby vsetky obrazky mali rovnaku priemernu uroven sivej farby
    X_train -= X_train.mean(axis=0)
    X_test -= X_test.mean(axis=0)
    
    # od kazdeho obrazku odcitame stredne hodnoty obrazkov, aby boli vycentrovane okolo pociatku suradnicovej sustavy
    X_train -= X_train.mean(axis=1).reshape(X_train.shape[0], -1)
    X_test -= X_test.mean(axis=1).reshape(X_test.shape[0], -1)
    
    return X_train, X_test
    
    
# funkcia na augmentaciu datovej sady
def augment_dataset(X, y):
    torch.manual_seed(17)
    
    # na kazdy obrazok vykoname nahodne prevratenie a nahodnu rotaciu
    transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.RandomRotation(degrees=(-20,20))])

    X_aug = []
    y_aug = []
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])

        # pre kazdy originalny obrazok vytvorime 5 novych obrazkov
        for j in range(5):
            X_aug.append(transform(X[i]))
            y_aug.append(y[i])
            
    return X_aug, y_aug
    
    
# funkcia na nacitanie datasetu
def load_dataset(base_path):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    ls = os.listdir(base_path)

    for d in ls:
        mode = d.split('_')[-1]
        tmp = d.split('_')[:-1]

        # ak ma priecinok v nazve podretazec 'train', tak budeme obrazky ukladat do zoznamu X_train, inak do X_test
        if mode == 'train':
            tmp_path = os.listdir(base_path + '/' + d)

            for img in tmp_path:
                if img.split('.')[-1] == 'png':
                    
                    # obrazok otvarame v grayscale mode
                    X_train.append(ImageOps.grayscale(Image.open(base_path + '/'+ d + '/' + img)))

                    # ak ma priecinok na zaciatku retazca podretazec 'target', tak sa jedna o obrazky detekovanej osoby
                    if tmp[0] == 'target':
                        y_train.append(1)
                    else:
                        y_train.append(0)
        else:
            tmp_path = os.listdir(base_path + '/' + d)

            for img in tmp_path:
                if img.split('.')[-1] == 'png':
                    
                    # obrazok otvarame v grayscale mode
                    X_test.append(ImageOps.grayscale(Image.open(base_path + '/'+ d + '/' + img)))

                    # ak ma priecinok na zaciatku retazca podretazec 'target', tak sa jedna o obrazky detekovanej osoby
                    if tmp[0] == 'target':
                        y_test.append(1)
                    else:
                        y_test.append(0)
                        
    return X_train, X_test, y_train, y_test
    
    
if __name__ == '__main__':
    # do premennej base_path treba zadat cestu k datasetu
    base_path = os.path.abspath('./dataset')
    X_train, X_test, y_train, y_test = load_dataset(base_path)
    
    # vykoname augmentaciu trenovacej sady, ako je vidiet tak niektore obrazky su zrotovane alebo prevratene
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train)
    
    # vykonavame predspracovanie obrazkov pre model RandomForest
    X_train_pre, X_test = preprocess_dataset(X_train_aug.copy(), X_test.copy())
    
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=17)
    clf = RandomForestClassifier(n_estimators=50, random_state=17)
    cv_results = cross_validate(clf, X_train_pre, y_train_aug, cv=cv, return_estimator=True)
    clf = cv_results['estimator'][np.argmax(cv_results['test_score'])]
    eval_model(os.path.abspath('./eval'), clf)  # tuto treba napisat cestu k evaluacnym datam
    