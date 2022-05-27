import matplotlib.pyplot as plt
import os
import copy
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from audiomentations import Compose, AddGaussianSNR, RoomSimulator, ClippingDistortion, LowShelfFilter, Shift, \
    PitchShift, HighShelfFilter, TimeStretch, HighPassFilter, PeakingFilter, LowPassFilter, AddGaussianNoise, \
    BandPassFilter, TimeMask, Reverse, BandStopFilter, FrequencyMask, TimeStretch, PitchShift, Shift, Gain
import malaya_speech
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
import argparse
import pickle

#autor: Michal Kabac(xkabac00)
#predmet: SUR
#rok: 2021/2022

#orezanie ticha z nahracok
def trim_silence_from_middle(signal, sampling_rate):
    signal_new = malaya_speech.astype.float_to_int(signal)  # data musia byt z floatu precastene do int
    audio = AudioSegment(signal_new.tobytes(), frame_rate=sampling_rate, sample_width=signal_new.dtype.itemsize,
                         channels=1)  # vytvorenie audiosegment chunku
    audio_chunks = split_on_silence(audio, min_silence_len=200, silence_thresh=-30,
                                    keep_silence=100)  # v pripade ze su v nahravke tiche miesta, odstranime ich
    signal_new = sum(audio_chunks)  # spoji vsetky chunky do 1 chunku
    signal_new = np.array(signal_new.get_array_of_samples())  # z formatu AudioSegment dostavam celociselne hodnoty
    signal_new = malaya_speech.astype.int_to_float(signal_new)  # hodnoty z intigerov prevediem na floaty

    return signal_new

#rozsirenie datovej sady
def augument_data(samples, sampling_rate, idx):
    if idx == 0:
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.012, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples

    elif idx == 1:
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.012, p=0.5),
            Gain(min_gain_in_db=-10, max_gain_in_db=30, p=0.5),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
            Reverse(p=1.0)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples

    elif idx == 2:
        augment = Compose([
            Gain(min_gain_in_db=-10, max_gain_in_db=30, p=0.5),
            BandStopFilter(p=0.85),
            # FrequencyMask(min_frequency_band=0.0,max_frequency_band = 0.4,p=0.85),
            BandPassFilter(p=0.5)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples

    elif idx == 3:
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.012, p=0.5),
            Gain(min_gain_in_db=-10, max_gain_in_db=30, p=0.5),
            BandStopFilter(p=0.5),
            # FrequencyMask(min_frequency_band=0.0,max_frequency_band = 0.4,p=0.5),
            BandPassFilter(p=0.5)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples

    elif idx == 4:
        augment = Compose([
            AddGaussianSNR(p=0.7, min_snr_in_db=30, max_snr_in_db=90),
            Gain(min_gain_in_db=-10, max_gain_in_db=30, p=0.5),
            # FrequencyMask(min_frequency_band=0.0,max_frequency_band = 0.4,p=0.5),
            BandStopFilter(p=0.5),
            BandPassFilter(p=0.5),
            Reverse(p=0.2)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples
    elif idx == 5:
        augment = Compose([
            AddGaussianSNR(p=1, min_snr_in_db=30, max_snr_in_db=94),
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples
    elif idx == 6:
        augment = Compose([
            Gain(min_gain_in_db=-15, max_gain_in_db=30, p=1),
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples
    elif idx == 7:
        augment = Compose([
            BandStopFilter(p=1),
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples
    elif idx == 8:
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1),
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples
    elif idx == 9:
        augment = Compose([
            PeakingFilter(p=1)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples
    elif idx == 10:
        augment = Compose([
            LowShelfFilter(p=1)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples
    elif idx == 11:
        augment = Compose([
            LowPassFilter(p=1)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples

    elif idx == 12:
        augment = Compose([
            HighPassFilter(p=1)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples

    elif idx == 13:
        augment = Compose([
            ClippingDistortion(p=1)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples
    elif idx == 14:
        augment = Compose([
            Shift(p=1)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples
    elif idx == 15:
        augment = Compose([
            PitchShift(p=1)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples
    elif idx == 16:
        augment = Compose([
            TimeStretch(p=1)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples
    elif idx == 17:
        augment = Compose([
            HighShelfFilter(p=1)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples
    elif idx == 18:
        augment = Compose([
            RoomSimulator(p=1.0)
        ])
        augmented_samples = augment(samples=samples, sample_rate=sampling_rate)
        return augmented_samples

#ziskanie priznakov pre kazdu nahravku v subore
def get_features_for_dir(directory, augumentation):
    speaker_features = []

    speaker_season_id = []

    hop_length = 400  # pri 16000 hertzoch by mal byt hop length 400 (ak som dobre pocital) 400~25ms

    # prechadzam jednotlivymi subormi v trenovacom priecinku
    for filename in os.listdir(directory):
        # filtrovanie zvukovych zaznamov s koncovkou wav
        if filename.endswith(".wav") and not filename.startswith("._"):
            speaker_season_id.append(filename[:7])

            signal, sampling_rate = malaya_speech.load(os.path.join(directory, filename))
            signal = librosa.effects.trim(signal, top_db=10)[0]
            signal = trim_silence_from_middle(signal, sampling_rate)
            if augumentation == True:
                for i in range(2):
                    for j in range(19):
                        new_data = augument_data(signal, sampling_rate, j)
                        mfcc_feat = librosa.feature.mfcc(y=new_data, sr=sampling_rate, hop_length=hop_length,
                                                         n_mfcc=13).T
                        speaker_features.append(copy.deepcopy(mfcc_feat))

            # vypocet mfcc priznakov
            mfcc_feat = librosa.feature.mfcc(y=signal, sr=sampling_rate, hop_length=hop_length, n_mfcc=13).T
            speaker_features.append(copy.deepcopy(mfcc_feat))

    return set(speaker_season_id), speaker_features

#ziskanie priznakov pre trenovacie dala (LDA nie je pouzite a nie je pouzite ani PCA - funckia bola najskor pouzita na to aky vplyv budu mat tieto metody)
def train_LDA(train_features_detect,train_features_not_detect):
    mfcc_feature,labels,scaler,pca = get_features(train_features_detect,train_features_not_detect,False)
    clf = []
    return clf,mfcc_feature,labels,scaler,pca

#vykreslenie grafu s mierami variabiliry pre MFCC priznaky
def draw_graph(pca):
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')
    plt.title('MFCC Explained Variance')
    plt.show()

#ziskanie priznakov
def get_features(train_features_detect, train_features_not_detect, use_pca):
    mean_features, std_features, var_features, labels = prepare_features(train_features_detect,
                                                                         train_features_not_detect)
    mfcc_features = np.concatenate((mean_features, std_features), axis=1)
    pca = []
    if use_pca:
        pca = PCA().fit(mfcc_features)
        draw_graph(pca)
        pca = PCA(n_components=7)
        mfcc_features = pca.fit_transform(mfcc_features)
    features = mfcc_features
    scaler = StandardScaler()

    mfcc_features = scaler.fit_transform(features)

    return mfcc_features, labels, scaler, pca

#priprava priznakov
def prepare_features(speaker_features_detect, speaker_features_not_detect):
    mean_features = []
    std_features = []
    var_features = []
    labels = []

    for i in range(len(speaker_features_not_detect)):
        stacked_features = np.vstack(speaker_features_not_detect[i])
        mean_mfccs = np.mean(stacked_features, axis=0)
        std_mfccs = np.std(stacked_features, axis=0)
        var_mfccs = np.var(stacked_features, axis=0)

        mean_features.append(mean_mfccs)
        std_features.append(std_mfccs)
        var_features.append(var_mfccs)

        labels.append(0)

    for i in range(len(speaker_features_detect)):
        stacked_features = np.vstack(speaker_features_detect[i])
        mean_mfccs = np.mean(stacked_features, axis=0)
        std_mfccs = np.std(stacked_features, axis=0)
        var_mfccs = np.var(stacked_features, axis=0)

        mean_features.append(mean_mfccs)
        std_features.append(std_mfccs)
        var_features.append(var_mfccs)

        labels.append(1)

    mean_features = np.array(mean_features)
    std_features = np.array(std_features)
    var_features = np.array(var_features)
    labels = np.array(labels)

    return mean_features, std_features, var_features, labels

#trenovanie SVM
def train_SVM(mfcc_features, labels):
    svm_linear = {'C': [0.1, 1, 10, 100],
                  'kernel': ['linear']}
    svm_others = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto'],
                  'kernel': ['poly', 'rbf', 'sigmoid']}
    parameters = [svm_linear, svm_others]
    model = GridSearchCV(SVC(probability=True), param_grid=parameters, cv=10)

    model.fit(mfcc_features, labels)
    get_model_stats(model)

    return model

#vypis najlepsieho modelu z trenovania
def get_model_stats(model):
    print(" Results from Random Search " )
    print("\n The best estimator across ALL searched params:\n",model.best_estimator_)
    print("\n The best score across ALL searched params:\n",model.best_score_)
    print("\n The best parameters across ALL searched params:\n",model.best_params_)

#ziskanie priznakov pre testovacie data
def prepare_features_for_predict(speaker_features):
    mean_features = []
    std_features = []
    var_features = []

    for i in range(len(speaker_features)):
        stacked_features = np.vstack(speaker_features[i])
        mean_mfccs = np.mean(stacked_features, axis=0)
        std_mfccs = np.std(stacked_features, axis=0)
        var_mfccs = np.var(stacked_features, axis=0)

        mean_features.append(mean_mfccs)
        std_features.append(std_mfccs)
        var_features.append(var_mfccs)

    mean_features = np.array(mean_features)
    std_features = np.array(std_features)
    var_features = np.array(var_features)

    return mean_features, std_features, var_features

#predikcia, ziskam tvrde rozhodnutie a na zaklade neho vyberiem s akou pravdepodobnostou si je model isty
def predict_RF(X_test,classifier):
    prediciton = classifier.predict(X_test)[0]
    probability = classifier.predict_proba(X_test)[0][1]
    return prediciton,probability

#ziskanie features pre testovacie data, ako aj predikcie detektora
def process_predict_features(directory, scaler, classifier):
    result_list = []
    hop_length = 400
    for filename in os.listdir(directory):
        # filtrovanie zvukovych zaznamov s koncovkou wav
        if filename.endswith(".wav") and not filename.startswith("._"):
            mean_features = []
            std_features = []
            signal, sampling_rate = malaya_speech.load(os.path.join(directory, filename))
            signal = librosa.effects.trim(signal, top_db=10)[0]
            signal = trim_silence_from_middle(signal, sampling_rate)
            mfcc_feat = librosa.feature.mfcc(y=signal, sr=sampling_rate, hop_length=hop_length, n_mfcc=13).T

            mean_mfccs = np.mean(mfcc_feat, axis=0)
            std_mfccs = np.std(mfcc_feat, axis=0)
            mean_features.append(mean_mfccs)
            std_features.append(std_mfccs)

            mean_features = np.array(mean_features)
            std_features = np.array(std_features)

            mfcc_features = np.concatenate((mean_features, std_features), axis=1)

            mfcc_features = scaler.transform(mfcc_features)

            result, confidence = predict_RF(mfcc_features, classifier)
            file_name = filename.split('.')[0]
            result_list.append((file_name, result, confidence))
    return result_list

#sparsovanie vstupnych argumentov
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory-train-detect', help='')
    parser.add_argument('--directory-train-not-detect', help='')
    parser.add_argument('--directory-val-detect', help='')
    parser.add_argument('--directory-val-not-detect', help='')
    parser.add_argument('--directory-eval', help='')
    parser.add_argument('--save-trained-model', help='')
    parser.add_argument('--path-save-trained-model', help='')
    parser.add_argument('--save-results-dir', help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.directory_train_detect is not None:
        directory_train_detect = args.directory_train_detect
    if args.directory_train_not_detect is not None:
        directory_train_not_detect = args.directory_train_not_detect
    if args.directory_val_detect is not None:
        directory_val_detect = args.directory_val_detect
    if args.directory_val_not_detect is not None:
        directory_val_not_detect = args.directory_val_not_detect
    if args.directory_eval is not None:
        directory_eval = args.directory_eval
    if args.save_trained_model is not None:
        save_trained_model = args.save_trained_model
    if args.path_save_trained_model is not None:
        path_save_trained_model = args.path_save_trained_model
    if args.save_results_dir is not None:
        save_results_dir = args.save_results_dir

    #nacitanie dat
    print("Loading data...")
    speakers_detect, train_features_detect = get_features_for_dir(directory_train_detect, False)
    speakers_not_detect, train_features_not_detect = get_features_for_dir(directory_train_not_detect, False)
    speakers_detect_val, train_features_detect_val = get_features_for_dir(directory_val_detect, False)
    speakers_detect_val_non, train_features_detect_val_non = get_features_for_dir(directory_val_not_detect, False)

    #spojenie trenovaacej a validacnej sady
    train_features_detect = train_features_detect + train_features_detect_val
    train_features_not_detect = train_features_not_detect + train_features_detect_val_non

    #trenovanie SVM
    print("Training model...")
    clf, mfcc_feature, labels, scaler, pca = train_LDA(train_features_detect, train_features_not_detect)
    classifier = train_SVM(mfcc_feature, labels)

    #ulozenie modelu
    if save_trained_model:
        pickle.dump(classifier, open(path_save_trained_model+'SVM.svm', 'wb'))

    #ziskanie predikcii
    print('Making prediction...')
    res = process_predict_features(directory_eval,scaler,classifier)

    #ulozenie vysledkov
    with open(save_results_dir, 'w') as f:
        for i in res:
            f.write(i[0] + ' ' + str(i[2]) + ' ' + str(i[1]) + '\n')


if __name__ == "__main__":
    main()
