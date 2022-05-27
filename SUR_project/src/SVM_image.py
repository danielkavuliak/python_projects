import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pickle
import textwrap

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate


# funkcia na evaluaciu modelu nad ostrymi datami
def eval_model(directory, clf1, pca_w, sc_w):
    file_name = []
    X = []
    
    images = os.listdir(directory)
    for img in images:
        if img.split('.')[-1] == 'png':
            X.append(Image.open(directory + '/' + img))
            file_name.append(img.split('.')[0])
            
    X, pca_w, sc_w = preprocess_dataset_2(X, pca_w, sc_w)
    probabilities = clf1.predict_proba(X)
    with open('SVM_image.txt', 'w') as f:
        is_detected = None
        
        for i in range(len(file_name)):
            if probabilities[i][1] >= 0.5:
                is_detected = 1
            else:
                is_detected = 0
                
            f.write(file_name[i] + ' ' + str(probabilities[i][1]) + ' ' + str(is_detected) + '\n')


# funkcia na predspracovanie datasetu
# v tejto funkcii vykonavame PCA analyzu nad obrazkami datasetu
def preprocess_dataset_2(X, pca_w=None, sc_w=None):
    X_pca = []
    
    # kazdy obrazok prekonvertujeme z PIL obrazku do tenzoru, nasledne do numpy pola a napokon spravime reshape do vektoru
    converter = transforms.PILToTensor()
    for i in range(len(X)):
        tmp = converter(X[i]).numpy()
        X_pca.append(tmp.reshape(tmp.shape[0] * tmp.shape[1] * tmp.shape[2]))
    
    X_pca = np.asarray(X_pca)
    
    # ak sme nevykonavali normalizaciu, tak vytvorime model na to, ktory sa nauci rozlozenie dat a vykoname ju
    # inak vykoname len normalizaciu
    if sc_w is None:
        sc_w = StandardScaler()
        X_pca = sc_w.fit_transform(X_pca)
    else:
        X_pca = sc_w.transform(X_pca)
        
    # ak sme nevykonavali PCA analyzu, tak vytvorime model, ktory sa nauci rozlozenie dat a zobrazia sa grafy pomerov variancii v priznakoch
    if pca_w is None:
        pca_w = PCA(n_components=50, random_state=17)
        pca_w.fit(X_pca)
        
    # z dat vyberieme 50 dimenzii
    X_pca = pca_w.transform(X_pca)
    
    return X_pca, pca_w, sc_w

    
# funkcia na augmentaciu datovej sady
def augment_dataset(X, y):
    torch.manual_seed(17)
    
    # nad obrazkami vykonavame nahodne prevratenie a nahodnu rotaciu
    transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.RandomRotation(degrees=(-20,20))])

    X_aug = []
    y_aug = []
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])

        # pre kazdy originalny obrazok, vytvorime 5 novych obrazkov
        for j in range(5):
            X_aug.append(transform(X[i]))
            y_aug.append(y[i])
            
    return X_aug, y_aug
    
    
# funkcia na nacitavanie datasetu
def load_dataset(base_path):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    ls = os.listdir(base_path)

    for d in ls:
        mode = d.split('_')[-1]
        tmp = d.split('_')[:-1]

        # ak sa nakonci retazca nachadza podretazec 'train', tak obrazky budeme ukladat do zoznamu X_train
        # inak do X_test
        if mode == 'train':
            tmp_path = os.listdir(base_path + '/' + d)

            for img in tmp_path:
                if img.split('.')[-1] == 'png':
                    X_train.append(Image.open(base_path + '/'+ d + '/' + img))

                    # ak sa na zaciatku retazca nachadza podretazec 'target', tak sa jedna o osobu, ktoru chceme detekovat
                    if tmp[0] == 'target':
                        y_train.append(1)
                    else:
                        y_train.append(0)
        else:
            tmp_path = os.listdir(base_path + '/' + d)

            for img in tmp_path:
                if img.split('.')[-1] == 'png':
                    X_test.append(Image.open(base_path + '/'+ d + '/' + img))

                    # ak sa na zaciatku retazca nachadza podretazec 'target', tak sa jedna o osobu, ktoru chceme detekovat
                    if tmp[0] == 'target':
                        y_test.append(1)
                    else:
                        y_test.append(0)
                        
    return X_train, X_test, y_train, y_test
    
    
if __name__ == '__main__':
    base_path = os.path.abspath('./dataset')                                                        # tu sa napise cesta k datovej sade
    X_train, X_test, y_train, y_test = load_dataset(base_path)
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train)                                    # vykonavame augmentaciu trenovacej datovej sady
    X_train_pca_2, pca_w, sc_w = preprocess_dataset_2(X_train_aug)
    X_test_pca_2, pca_w, sc_w = preprocess_dataset_2(X_test, pca_w, sc_w)
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=17)
    clf1 = SVC(max_iter=10, random_state=17, probability=True)
    cv_results1 = cross_validate(clf1, X_train_pca_2, y_train_aug, cv=cv, return_estimator=True)    # krizova validacia pre model
    clf1 = cv_results1['estimator'][np.argmax(cv_results1['test_score'])]
    eval_model(os.path.abspath('./eval'), clf1, pca_w, sc_w)                                                           # do argumentu napiseme cestu k evaluacnym datam