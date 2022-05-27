from audiomentations import Compose, AddGaussianSNR, RoomSimulator, ClippingDistortion, LowShelfFilter, Shift, \
    PitchShift, HighShelfFilter, TimeStretch, HighPassFilter, PeakingFilter, LowPassFilter, AddGaussianNoise, \
    BandPassFilter, TimeMask, Reverse, BandStopFilter, FrequencyMask, TimeStretch, PitchShift, Shift, Gain
import malaya_speech
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
import sklearn.mixture
import matplotlib.pyplot as plt
import sklearn.mixture
import os
import copy
import numpy as np
import argparse
import pickle

#autor: Michal Kabac(xkabac00)
#predmet: SUR
#rok: 2021/2022

# rozsirenie datovej sady
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


#orezanie ticha z nahravok
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

#predikcia skore pre GMM, vypocitanie pomeru dvoch GMM a rozhodnutie
def predict_score(gmm, gmm_not_detect, speaker_features, hranica):
    confidence = None
    result = None
    is_detected = gmm.score(speaker_features)
    not_detected = gmm_not_detect.score(speaker_features)
    # print("likelihood detekovany ",(gmm, is_detected))
    # print("likelihood nedetekovany ",(gmm_not_detect,not_detected))
    decision = is_detected / not_detected
    # print(decision)
    if decision < hranica:
        result = 1
    else:
        result = 0

    confidence = hranica - decision
    return result, confidence

#trenovanie GMM
def train_gmm(train_features,n_components):
    #trenovanie gmm modelu s n komponentmi
    stacked_features = np.vstack(train_features)
    #gmm = BayesianGaussianMixture(n_components=n_components)
    gmm = sklearn.mixture.GaussianMixture(n_components=n_components)
    gmm.fit(stacked_features)
    return gmm

#predikcia pre zaznami v subore
def predict_features_for_dir(directory, gmm, gmm_not_detect,hranica):
    result_list = []
    hop_length = 400
    for filename in os.listdir(directory):
        # filtrovanie zvukovych zaznamov s koncovkou wav
        if filename.endswith(".wav") and not filename.startswith("._"):
            signal, sampling_rate = malaya_speech.load(os.path.join(directory, filename))
            signal = librosa.effects.trim(signal, top_db=10)[0]
            signal = trim_silence_from_middle(signal, sampling_rate)
            mfcc_feat = librosa.feature.mfcc(y=signal, sr=sampling_rate, hop_length=hop_length, n_mfcc=13).T
            result, confidence = predict_score(gmm, gmm_not_detect, mfcc_feat, hranica)
            file_name = filename.split('.')[0]
            result_list.append((file_name, result, confidence))
    return result_list

#evaulvacia skore(v pripade data augumentation)
def eval_score(gmm, gmm_not_detect, speaker_features):
    confidence = None
    result = None
    is_detected = gmm.score(speaker_features)
    not_detected = gmm_not_detect.score(speaker_features)
    # print("likelihood detekovany ",(gmm, is_detected))
    # print("likelihood nedetekovany ",(gmm_not_detect,not_detected))
    decision = is_detected / not_detected
    return decision

#evaulvacia skore pre subor(v pripade data augumentation)
def eval_features_for_dir(directory, gmm, gmm_not_detect):
    result_list = []
    hop_length = 400
    for filename in os.listdir(directory):
        # filtrovanie zvukovych zaznamov s koncovkou wav
        if filename.endswith(".wav") and not filename.startswith("._"):
            signal, sampling_rate = malaya_speech.load(os.path.join(directory, filename))
            signal = librosa.effects.trim(signal, top_db=10)[0]
            signal = trim_silence_from_middle(signal, sampling_rate)
            mfcc_feat = librosa.feature.mfcc(y=signal, sr=sampling_rate, hop_length=hop_length, n_mfcc=13).T
            result = eval_score(gmm, gmm_not_detect, mfcc_feat)
            result_list.append(result)
    return result_list

#ziskanie priznakov pre subor
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
                for i in range(1):
                    for j in range(19):
                        new_data = augument_data(signal, sampling_rate, j)
                        mfcc_feat = librosa.feature.mfcc(y=new_data, sr=sampling_rate, hop_length=hop_length,
                                                         n_mfcc=13).T
                        speaker_features.append(copy.deepcopy(mfcc_feat))

            # vypocet mfcc priznakov
            mfcc_feat = librosa.feature.mfcc(y=signal, sr=sampling_rate, hop_length=hop_length, n_mfcc=13).T
            speaker_features.append(copy.deepcopy(mfcc_feat))

    return set(speaker_season_id), speaker_features

#parsovanie vstupnych argumentov
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


    augumentation_data = True #ak nechceme augumentovat data nastavime na false

    #nacitavanie dat
    print("Start loading data...")
    speakers_detect, train_features_detect = get_features_for_dir(directory_train_detect, augumentation_data)
    speakers_not_detect, train_features_not_detect = get_features_for_dir(directory_train_not_detect, augumentation_data)

    #spojenie trenovacej a validacnej sady
    if not augumentation_data:
        speakers_detect_val, train_features_detect_val = get_features_for_dir(directory_val_detect, False)
        speakers_detect_val_non, train_features_detect_val_non = get_features_for_dir(directory_val_not_detect, False)
        train_features_detect = train_features_detect + train_features_detect_val
        train_features_not_detect = train_features_not_detect + train_features_detect_val_non

    #trenovanie GMM pre hladanu a pre ostatne osoby
    print("Start training model")
    gmm_detect_person = train_gmm(train_features_detect, 15)
    gmm_not_detect_person = train_gmm(train_features_not_detect, 15)

    if augumentation_data:
        result_detect = eval_features_for_dir(directory_val_detect, gmm_detect_person, gmm_not_detect_person)
        result_not_detect = eval_features_for_dir(directory_val_not_detect, gmm_detect_person, gmm_not_detect_person)

    #ulozenie modelu
    if save_trained_model:
        print("Saving model")
        pickle.dump(gmm_detect_person, open(path_save_trained_model+'gmm_detect.gmm', 'wb'))
        pickle.dump(gmm_not_detect_person, open(path_save_trained_model+'gmm_not_detect.gmm', 'wb'))

    # predikcie modelu
    print("Making predictions")
    if augumentation_data:
        print(f"Decision calculated by values {max(result_detect)}, {min(result_not_detect)}")
        hranica = (max(result_detect)+min(result_not_detect))/2
        print(hranica)
        res = predict_features_for_dir(directory_eval, gmm_detect_person, gmm_not_detect_person,hranica)
    else:
        res = predict_features_for_dir(directory_eval, gmm_detect_person, gmm_not_detect_person,1)

    #zapis vysledkov do suboru
    with open(save_results_dir, 'w') as f:
        for i in res:
            f.write(i[0] + ' ' + str(i[2]) + ' ' + str(i[1]) + '\n')


if __name__ == "__main__":
    main()
