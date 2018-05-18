# -*- coding: utf-8 -*-
import numpy as np
import pyedflib
import scipy.io

from eegpredict.functions import globals
from eegpredict.functions.utils import EmptyClass
from stacklineplot import stackplot, stackplot_my
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler, MaxAbsScaler

from eegpredict.functions.utils import score_svm_coefficients, moving_average

subtitle_len = 250


def plotTestData(fold_pairs,fold, pat_id, feature_info, window_length=5):
    pre_fold_idx=fold_pairs[fold][4][0]
    int_fold_idx = fold_pairs[fold][4][1]

    # plot preictal eeg
    idx = feature_info[5][0][0][0]-1    #feature icerisinde yeri
    file = feature_info[5][0][0][1]
    start = np.squeeze(feature_info[5][0][0][2]) #eeg record icinde basi
    end = np.squeeze(feature_info[5][0][0][3])  # eeg record icinde sonu
    seizureList = np.squeeze(scipy.io.loadmat(globals.seizureList_file)['seizureList'])

    sph=5*60
    max_s_length=15*60
    aidx = np.squeeze(np.asarray(np.where(np.squeeze((idx > pre_fold_idx[0]) & (idx < pre_fold_idx[1])))))
    aidx_first = np.asarray(np.where(np.squeeze(idx <= pre_fold_idx[0])))[0,-1]
    aidx = np.array(np.insert(aidx, 0, aidx_first), dtype=int)
    sei_start=[]
    sei_end=[]
    sig_all= np.array([])
    for i in aidx:
        # first file
        if i==aidx[0]:
            start_idx=start[i]+(pre_fold_idx[0]-np.int(idx[i]))*window_length
        else:
            start_idx=start[i]
        # last file
        if i==aidx[-1]:
            end_idx_sph = start_idx + (pre_fold_idx[1] - np.int(idx[i])) * window_length
            end_idx = end_idx_sph + sph + max_s_length
            for s in range(seizureList.shape[0]):
                if np.str(seizureList[s][0][0]) == np.str(file[i][0][0]):
                    sei_start.append(np.int(seizureList[s][2][0]))
                    sei_end.append(np.int(seizureList[s][3][0]))

        else:
            end_idx=end[i]

        # first file
        if i == aidx[0]:
            start_idx_pre=start_idx-end_idx

        f = pyedflib.EdfReader(globals.dataset_link + np.str(file[i][0][0]))
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()

        fs=np.int(f.getSampleFrequencies()[0])
        if (end_idx*fs) >f.getNSamples()[0]:
            end_idx=f.getNSamples()[0]/fs
        sigbufs = np.zeros((n, (end_idx-start_idx)*fs))
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i, start=start_idx*fs, n=(end_idx-start_idx)*fs)

        f._close()
        del f

        sig_all = np.concatenate([sig_all, sigbufs],1) if sig_all.size else sigbufs

    if aidx.shape[0] > 1:
        start_idx = start_idx_pre

    fig1 = stackplot(sig_all, seconds=(end_idx-start_idx), start_time=start_idx, ylabels=signal_labels, show=True)
    fig1.axvline(x=sei_start, color='red', linewidth=1.5, label='seizure onset')
    fig1.axvspan(xmin=sei_start[0]-sph, xmax=sei_start[0], facecolor='gray', label='SPH')
    #fig.axvline(x=sei_end)
    fig1.grid()
    fig1.legend()
    fig1.set_title('Patient {0} - Seizure {1} Preictal Record'.format(pat_id, fold))

    # plot interictal eeg
    idx = feature_info[4][0][0][0] - 1  # feature icerisinde yeri
    file = feature_info[4][0][0][1]
    start = np.squeeze(feature_info[4][0][0][2])  # eeg record icinde basi
    end = np.squeeze(feature_info[4][0][0][3])  # eeg record icinde sonu

    aidx = np.squeeze(np.asarray(np.where(np.squeeze((idx > int_fold_idx[0]) & (idx < int_fold_idx[1])))))
    aidx_first = np.asarray(np.where(np.squeeze((idx <= int_fold_idx[0]))))[0,-1]
    aidx= np.array(np.insert(aidx,0,aidx_first),dtype=int)
    sig_all = np.array([])
    part_end = np.array([])
    for i in aidx:
        # first file
        if i == aidx[0]:
            start_idx = start[i] + (int_fold_idx[0] - np.int(idx[i])) * window_length
        else:
            start_idx = start[i]
        # last file
        if i == aidx[-1]:
            end_idx = start_idx + (int_fold_idx[1] - np.int(idx[i])) * window_length
        else:
            end_idx = end[i]

        # first file
        if i == aidx[0]:
            start_idx_pre = start_idx - end_idx

        f = pyedflib.EdfReader(globals.dataset_link + np.str(file[i][0][0]))
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()

        fs = np.int(f.getSampleFrequencies()[0])
        if (end_idx * fs) > f.getNSamples()[0]:
            end_idx = f.getNSamples()[0] / fs
        sigbufs = np.zeros((n, (end_idx - start_idx) * fs))
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i, start=start_idx * fs, n=(end_idx - start_idx) * fs)

        f._close()
        del f

        part_len = end_idx - start_idx
        part_end = np.concatenate([part_end, np.asarray([part_end[-1]+part_len])], 0) if part_end.size else np.array([part_len])
        sig_all = np.concatenate([sig_all, sigbufs], 1) if sig_all.size else sigbufs
    fig2 = stackplot(sig_all, seconds=part_end[-1], start_time=0, ylabels=signal_labels, show=True)
    for i in range(part_end.shape[0]):
        fig2.axvline(x=part_end[i], color='blue', linewidth=0.5, label=np.str(file[aidx[i]][0][0]))
    fig2.legend()
    fig2.set_title('Patient {0} - Seizure {1} Interictal Record'.format(pat_id, fold))

def plotTestDataNew(fold_pairs, fold, pat_id, feature_info, window_length=5):
    pre_fold_idx = fold_pairs[fold][4][0]
    int_fold_idx = fold_pairs[fold][4][1]

    # plot preictal eeg
    idx = feature_info[5][0][0][0]  # feature icerisinde yeri
    file = feature_info[5][0][0][1]
    start = np.squeeze(feature_info[5][0][0][2])  # eeg record icinde basi
    end = np.squeeze(feature_info[5][0][0][3])  # eeg record icinde sonu
    seizureList = np.squeeze(scipy.io.loadmat(globals.seizureList_file)['seizureList'])

    seizure_feats = np.squeeze(feature_info[3])
    seizure_feats = np.append(seizure_feats,feature_info[1])
    pre_idx =[seizure_feats[pre_fold_idx],seizure_feats[pre_fold_idx+1]]
    sph = 5 * 60
    max_s_length = 15 * 60
    aidx = np.squeeze(np.asarray(np.where(np.squeeze((idx > pre_idx[0]) & (idx < pre_idx[1])))))
    aidx_first = np.asarray(np.where(np.squeeze(idx <= pre_idx[0])))[0, -1]
    aidx = np.array(np.insert(aidx, 0, aidx_first), dtype=int)
    sei_start = []
    sei_end = []
    sig_all = np.array([])
    part_end = np.array([])
    for i in aidx:
        # first file
        if i == aidx[0]:
            start_idx = start[i] + (pre_idx[0] - np.int(idx[i])) * window_length
        else:
            start_idx = start[i]
        # last file
        if i == aidx[-1]:
            end_idx_sph = start_idx + (pre_idx[1] - np.int(idx[i])) * window_length
            end_idx = end_idx_sph + sph + max_s_length
            for s in range(seizureList.shape[0]):
                if np.str(seizureList[s][0][0]) == np.str(file[i][0][0]):
                    sei_start.append(np.int(seizureList[s][2][0]))
                    sei_end.append(np.int(seizureList[s][3][0]))

        else:
            end_idx = end[i]

        # first file
        if i == aidx[0]:
            start_idx_pre = start_idx - end_idx

        f = pyedflib.EdfReader(globals.dataset_link + np.str(file[i][0][0]))
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()

        fs = np.int(f.getSampleFrequencies()[0])
        if (end_idx * fs) > f.getNSamples()[0]:
            end_idx = f.getNSamples()[0] / fs
        sigbufs = np.zeros((n, (end_idx - start_idx) * fs))
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i, start=start_idx * fs, n=(end_idx - start_idx) * fs)

        f._close()
        del f

        part_len = end_idx - start_idx
        part_end = np.concatenate([part_end, np.asarray([part_end[-1] + part_len])],
                                  0) if part_end.size else np.array([part_len])

        sig_all = np.concatenate([sig_all, sigbufs], 1) if sig_all.size else sigbufs

    if aidx.shape[0] > 1:
        start_idx = start_idx_pre

    fig1 = stackplot(sig_all, seconds=(end_idx - start_idx), start_time=start_idx, ylabels=signal_labels, show=True, dataRange=[-1024*0.7, 1024*0.7])
    colors = iter(cm.rainbow(np.linspace(0, 1, part_end.shape[0])))
    part_end = np.insert(part_end, 0, 0)
    for i in range(part_end.shape[0]-1):
        curcolor = next(colors)
        fig1.axhspan(fig1.get_ylim()[1]-subtitle_len, fig1.get_ylim()[1],float(part_end[i])/part_end[-1], float(
            part_end[i+1])/part_end[-1],color=curcolor, label=np.str(file[aidx[i]][0][0]))
    fig1.axvline(x=sei_start, color='red', linewidth=1.5, label='nobet baslangici')
    fig1.axvspan(xmin=sei_start[0] - sph, xmax=sei_start[0], facecolor='gray', label='SPH')
    fig1.legend(loc='lower right', bbox_to_anchor=(1, 0),
          fancybox=False, shadow=False)
    fig1.set_title('Hasta {0} - Nobet {1} Preictal Kaydi'.format(pat_id, fold))

    # plot interictal eeg
    idx = feature_info[4][0][0][0] # feature icerisinde yeri
    file = feature_info[4][0][0][1]
    start = np.squeeze(feature_info[4][0][0][2])  # eeg record icinde basi
    end = np.squeeze(feature_info[4][0][0][3])  # eeg record icinde sonu

    sig_all = np.array([])
    part_end = np.array([])
    for i in int_fold_idx:

        start_idx = start[i]
        end_idx = end[i]

        f = pyedflib.EdfReader(globals.dataset_link + np.str(file[i][0][0]))
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()

        fs = np.int(f.getSampleFrequencies()[0])
        if (end_idx * fs) > f.getNSamples()[0]:
            end_idx = f.getNSamples()[0] / fs
        sigbufs = np.zeros((n, (end_idx - start_idx) * fs))
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i, start=start_idx * fs, n=(end_idx - start_idx) * fs)
        f._close()
        del f

        part_len = end_idx - start_idx
        part_end = np.concatenate([part_end, np.asarray([part_end[-1] + part_len])],
                                  0) if part_end.size else np.array([part_len])
        sig_all = np.concatenate([sig_all, sigbufs], 1) if sig_all.size else sigbufs
    fig2 = stackplot(sig_all, seconds=part_end[-1], start_time=0, ylabels=signal_labels, show=True, dataRange=[-1024*0.7, 1024*0.7])
    colors = iter(cm.rainbow(np.linspace(0, 1, part_end.shape[0])))
    part_end=np.insert(part_end, 0, 0)
    for i in range(part_end.shape[0]-1):
        curcolor = next(colors)
        fig2.axvline(x=part_end[i+1], color=curcolor, linewidth=0.75)
        fig2.axhspan(fig2.get_ylim()[1]-subtitle_len, fig2.get_ylim()[1],float(part_end[i])/part_end[-1], float(
            part_end[i+1])/part_end[-1],color=curcolor, label=np.str(file[int_fold_idx[i]][0][0]))
    fig2.legend(loc='lower right', bbox_to_anchor=(1, 0),
          fancybox=False, shadow=False)
    fig2.set_title('Patient {0} - Seizure {1} Interictal Record'.format(pat_id, fold))

def plotFeatureData(feature_data, fold_pairs, fold, pat_id, feature_info):
    plt.imshow(feature_data, extent=[0, 1, 0, 1])

def get_eeg_features(dataset, preictal_idx, interictal_idx, pca=True, component=3, scaler=None, with_mean=True, channel_count=20):

    # preictal ve interictal parcalarin dizi icerisindeki indeksleri
    pre_feature_idx = np.asarray(dataset.feature_info[3])-1
    pre_feature_idx = np.append(pre_feature_idx, dataset.feature_info[1]+1)

    int_feature_idx = np.asarray(dataset.feature_info[2])-1
    int_feature_idx = np.append(int_feature_idx, dataset.feature_info[0]+1)

    # features
    feature_type_idx=dataset.feat_params.feature_eval * channel_count
    feature_type_idx_range=np.hstack(range(i, i+channel_count) for i in feature_type_idx)

    # preictal features
    feature_pre = scipy.io.loadmat('../../generateEEGFeats/dataset/pat_' + str(dataset.pat_id) + '_pre.mat')['feature']
    feature_pre = np.nan_to_num(np.asarray(feature_pre[:, 0:-2]))
    feature_pre = feature_pre[:, feature_type_idx_range]
    # interictal features
    feature_int = scipy.io.loadmat('../../generateEEGFeats/dataset/pat_' + str(dataset.pat_id) + '_int.mat')['feature']
    feature_int = np.nan_to_num(np.asarray(feature_int[:, 0:-1]))
    feature_int = feature_int[:, feature_type_idx_range]

    if scaler == "standard":
        sc = StandardScaler(with_mean=with_mean)
    elif scaler == "robust":
        sc = RobustScaler(with_centering=with_mean)
    elif scaler == "normalize":
        sc = Normalizer()
    elif scaler == "maxabs":
        sc = MaxAbsScaler()
    elif scaler == "minmax":
        sc = MinMaxScaler()

    # preictal ve interictal veri tum veriye gore scale ediliyor
    if scaler:
        sc.fit(np.concatenate([feature_pre, feature_int]))
        feature_pre = sc.transform(feature_pre)
        feature_int = sc.transform(feature_int)

    feature_preictal = np.nan_to_num(feature_pre[pre_feature_idx[preictal_idx]:pre_feature_idx[preictal_idx+1]])
    feature_interictal = []
    for i in interictal_idx:
        feature_interictal.append(feature_int[int_feature_idx[i]:int_feature_idx[i+1]])
    feature_interictal_all=feature_interictal[0]
    for i in range(1, len(feature_interictal)):
        feature_interictal_all = np.concatenate([feature_interictal_all, feature_interictal[i]])

    # degisintinin en cok oldugu pca bilesenleri bulunuyor
    if pca:
        pca_ = PCA(n_components=component)
        pca_.fit(np.concatenate([feature_preictal, feature_interictal_all]))
        feature_preictal = pca_.transform(feature_preictal)
        for i in range(len(feature_interictal)):
            feature_interictal[i] = pca_.transform(feature_interictal[i])
        feature_idx = range(component)

        labels = []
        for i in feature_idx:
            labels.append("pc:" + str(i + 1))

    # en belirleyici bilesenler svm ile bulunuyor
    else:
        svc = svm.SVC(kernel='linear', class_weight='balanced', probability=False)
        svc.fit(np.concatenate([feature_preictal, feature_interictal[0]]),
                np.concatenate([np.ones((len(feature_preictal))), np.zeros((len(feature_interictal[0])))]))
        # svc.fit(np.concatenate([feature_preictal, feature_interictal_all]),
        #         np.concatenate([np.ones((len(feature_preictal))), np.zeros((len(feature_interictal_all)))]))
        feature_idx, _, labels = score_svm_coefficients(svc, asType="all", top_features=component, show=False)
        feature_preictal = feature_preictal[:, feature_idx]
        for i in range(len(feature_interictal)):
            feature_interictal[i] = feature_interictal[i][:, feature_idx]

    return feature_preictal, feature_interictal, feature_idx, labels


def read_eeg_signal(file_name, start_idx, end_idx):

    f = pyedflib.EdfReader(globals.dataset_link + np.str(file_name))
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()

    fs = np.int(f.getSampleFrequencies()[0])
    if not end_idx:
        end_idx = f.getNSamples()[0] / fs
    if (end_idx * fs) > f.getNSamples()[0]:
        end_idx = f.getNSamples()[0] / fs
    signal = np.zeros((n, (end_idx - start_idx) * fs))
    for i in np.arange(n):
        signal[i, :] = f.readSignal(i, start=start_idx * fs, n=(end_idx - start_idx) * fs)

    f._close()
    del f

    return signal, signal_labels


def get_eeg_signals(dataset, preictal_idx, interictal_idx, get_ictal=True):
    preictal_signal = EmptyClass()
    # preictal signal
    pre_idx = np.squeeze(dataset.feature_info[3]) # preictal fold indexi
    pre_idx = np.append(pre_idx, dataset.feature_info[1]) # preictal fold indexine son index ekleniyor
    pre_part_idx = dataset.feature_info[5][0][0][0]
    pre_part_idx = np.append(pre_part_idx, dataset.feature_info[1])
    pre_part_file = dataset.feature_info[5][0][0][1] # preictal parcanin dosya ismi
    pre_part_start = np.squeeze(dataset.feature_info[5][0][0][2])  # eeg record icinde basi
    pre_part_end = np.squeeze(dataset.feature_info[5][0][0][3])  # eeg record icinde sonu

    pre_idx_start = pre_idx[preictal_idx]
    pre_idx_end = pre_idx[preictal_idx+1]
    pre_fold_idx = np.where((pre_part_idx >= pre_idx_start) & (pre_part_idx < pre_idx_end))[0]

    preictal_signal.file = pre_part_file[pre_fold_idx]
    preictal_signal.signal = np.array([])
    preictal_signal.signal_idx = [0]
    preictal_signal.feat_idx_start = pre_part_idx[pre_fold_idx]
    preictal_signal.feat_idx_end = pre_part_idx[pre_fold_idx+1]
    for i in pre_fold_idx:
        signal, signal_labels = read_eeg_signal(pre_part_file[i][0][0], pre_part_start[i], pre_part_end[i])
        preictal_signal.signal = np.concatenate([preictal_signal.signal, signal], 1) if preictal_signal.signal.size else signal
        preictal_signal.signal_idx.append(preictal_signal.signal_idx[-1]+signal.shape[1])
        pre_fold_end=pre_part_end[i]

    interictal_signal = EmptyClass()
    # interictal signal
    int_part_idx = dataset.feature_info[4][0][0][0]  # interictal parcanin feature icerisinde yeri
    int_part_idx = np.append(int_part_idx, dataset.feature_info[0])
    int_part_file = dataset.feature_info[4][0][0][1]  # interictal parcanin dosya ismi
    int_part_start = np.squeeze(dataset.feature_info[4][0][0][2])  # eeg record icinde basi
    int_part_end = np.squeeze(dataset.feature_info[4][0][0][3])  # eeg record icinde sonu

    interictal_signal.file = int_part_file[interictal_idx]
    interictal_signal.signal = np.array([])
    interictal_signal.signal_idx = [0]
    interictal_signal.feat_idx_start = int_part_idx[interictal_idx]
    interictal_signal.feat_idx_end = int_part_idx[interictal_idx+1]
    for i in interictal_idx:
        signal, signal_labels = read_eeg_signal(int_part_file[i][0][0], int_part_start[i], int_part_end[i])
        interictal_signal.signal = np.concatenate([interictal_signal.signal, signal], 1) if interictal_signal.signal.size else signal
        interictal_signal.signal_idx.append(interictal_signal.signal_idx[-1]+signal.shape[1])

    if get_ictal:
        seizureList = np.squeeze(scipy.io.loadmat(globals.seizureList_file)['seizureList'])
        recordList = np.squeeze(scipy.io.loadmat(globals.records_file)['records'])
        seizureListFiles = [seizureList[i][0][0] for i in range(len(seizureList))]
        recordListFiles = [recordList[i][0][0][0][0] for i in range(len(recordList))]

        ictal_signal = EmptyClass()
        ictal_signal.signal_idx = [0]
        # preictal ile ictal ayni dosyada mi
        if preictal_signal.file[-1][0][0] in seizureListFiles:
            ictal_idx=np.where(np.array(seizureListFiles) == preictal_signal.file[-1][0][0])[0][0]
            ictal_part_start = seizureList[ictal_idx][2][0][0]
            ictal_part_end = seizureList[ictal_idx][3][0][0]

            ictal_signal.file = [preictal_signal.file[-1][0][0], preictal_signal.file[-1][0][0]]
            # Varsa SPH Kismi
            ictal_signal.signal_idx = [0]
            if pre_fold_end < ictal_part_start:
                signal, signal_labels = read_eeg_signal(ictal_signal.file[0], pre_fold_end, ictal_part_start)
                ictal_signal.signal = signal
                ictal_signal.signal_idx.append(ictal_signal.signal_idx[-1]+signal.shape[1])
            # Ictal Kismi
            signal, signal_labels = read_eeg_signal(ictal_signal.file[1], ictal_part_start, ictal_part_end)
            ictal_signal.signal = np.concatenate([ictal_signal.signal, signal], 1) if ictal_signal.signal.size else signal
            ictal_signal.signal_idx.append(ictal_signal.signal_idx[-1] + signal.shape[1])
        else:
            ictal_idx = np.where(np.array(recordListFiles) == preictal_signal.file[-1][0][0])
            ictal_idx = np.where(np.array(seizureListFiles) == recordListFiles[ictal_idx[0][0]+1])[0][0]
            ictal_part_start = seizureList[ictal_idx][2][0][0]
            ictal_part_end = seizureList[ictal_idx][3][0][0]
            ictal_signal.file = [preictal_signal.file[-1][0][0], seizureListFiles[ictal_idx]]

            ictal_signal.signal_idx = [0]
            # SPH Kismi
            # dosya preictal sonundan baslanarak sonuna kadar okunuyor
            signal, signal_labels = read_eeg_signal(ictal_signal.file[0], pre_fold_end, [])
            ictal_signal.signal = signal
            ictal_signal.signal_idx.append(ictal_signal.signal_idx[-1] + signal.shape[1])
            # sph'in diger kismi bir sonraki dosyadan okunuyor
            signal, signal_labels = read_eeg_signal(ictal_signal.file[1], 0, ictal_part_start)
            ictal_signal.signal = np.hstack((ictal_signal.signal, signal))
            ictal_signal.signal_idx[-1] = ictal_signal.signal_idx[-1] + signal.shape[1]
            # Ictal Kismi
            signal, signal_labels = read_eeg_signal(ictal_signal.file[1], ictal_part_start, ictal_part_end)
            ictal_signal.signal = np.concatenate([ictal_signal.signal, signal],
                                                 1) if ictal_signal.signal.size else signal
            ictal_signal.signal_idx.append(ictal_signal.signal_idx[-1] + signal.shape[1])
    return preictal_signal, interictal_signal, ictal_signal, signal_labels


def plot_eeg_prediction(dataset, predict_result, window_length=4, overlap=0, fs=256, include_ictal=True):

    window_length =window_length*(1-overlap)

    seizure_count = len(dataset.fold_pairs)
    fold_count = dataset.num_fold

    for seizure in range(seizure_count):

        # preictal ve intercital prediction verileri sadelestiriliyor
        preictal_pred = np.concatenate([np.array(i) for i in predict_result.swapaxes(1, 2)[seizure, 0]]).swapaxes(0, 1).mean(axis=1)
        interictal_pred = np.concatenate([np.array(i) for i in predict_result.swapaxes(1, 2)[seizure, 1]]).swapaxes(0, 1).mean(axis=1)

        # test icin kullanilan preictal ve interictal parcalarin indeksleri
        preictal_idx = dataset.fold_pairs[seizure][4][0]
        interictal_idx = dataset.fold_pairs[seizure][4][1]

        # test icin kullanilan preictal ve interictal parcalara denk dusen eeg featurelari
        test_preictal_feats, test_interictal_feats, feat_idx, labels = get_eeg_features(dataset, preictal_idx,
                                                                                        interictal_idx, pca=False,
                                                                                        scaler="standard",
                                                                                        component=3)

        # test icin kullanilan preictal ve interictal parcalara denk dusen eeg sinyalleri
        preictal_signal, interictal_signal, ictal_signal, signal_labels = get_eeg_signals(dataset, preictal_idx, interictal_idx)

        # grafikler olusturuluyor
        # plot preictal
        fig = plt.figure()
        gs = gridspec.GridSpec(3, 1, height_ratios=[4, 2, 1])
        axes_0 = plt.subplot(gs[0])
        axes_2 = plt.subplot(gs[1])
        axes_1 = plt.subplot(gs[2])


        fig.suptitle('Patient {0} - Seizure {1}'.format(dataset.pat_id, seizure+1), fontweight='bold')

        if include_ictal:
            # preictal ve ictal beraber cizdiriliyor
            signal = np.hstack((preictal_signal.signal, ictal_signal.signal))
            signal_start = preictal_signal.signal_idx[0]
            signal_end = preictal_signal.signal_idx[-1] + ictal_signal.signal_idx[-1]

            part_count = len(preictal_signal.signal_idx) + len(ictal_signal.signal_idx) - 2
            colors = iter(cm.rainbow(np.linspace(0, 1, part_count)))
        else:
            # preictal cizdiriliyor
            signal = preictal_signal.signal
            signal_start = preictal_signal.signal_idx[0]
            signal_end = preictal_signal.signal_idx[-1]

            part_count = len(preictal_signal.signal_idx) - 1
            colors = iter(cm.rainbow(np.linspace(0, 1, part_count)))

        axes_0 = stackplot_my(signal,
                            seconds=(signal_end - signal_start)/fs,
                            start_time=signal_start/fs, ylabels=signal_labels,
                            dataRange=[-1024 * 0.5, 1024 * 0.5], my_figure=axes_0, channels=[])
        axes_0.set_title('Preictal EEG Signal')

        # cikarilan featurelar
        test_preictal_feat = test_preictal_feats.swapaxes(0, 1)
        test_preictal_feat = np.asarray([moving_average(i, 3) for i in test_preictal_feat]).swapaxes(0, 1)
        axes_2.plot(test_preictal_feat)
        axes_2.legend(labels)
        axes_2.set_xlim(0, (signal_end/fs)/window_length)
        #axes_2.set_ylim(-3, 3)
        axes_2.set_title('Preictal Features')


        # prediction sonuclari
        preictal_pred_avg = moving_average(preictal_pred, 3)
        axes_1.plot(preictal_pred_avg)
        axes_1.set_xlim(0, (signal_end/fs)/window_length)
        axes_1.set_ylim(-0.1, 1.1)
        axes_1.set_title('Predictions')

        #once preictal kisim
        for i in range(len(preictal_signal.signal_idx) - 1):
            curcolor = next(colors)
            axes_0.axhspan(axes_0.get_ylim()[1] - subtitle_len, axes_0.get_ylim()[1],
                           float(preictal_signal.signal_idx[i]) / signal_end,
                           float(preictal_signal.signal_idx[i + 1]) / signal_end, color=curcolor,
                           label=np.str(preictal_signal.file[i][0][0]))
            axes_1.axhspan(axes_1.get_ylim()[1] - 0.05, axes_1.get_ylim()[1],
                           float(preictal_signal.signal_idx[i]) / signal_end,
                           float(preictal_signal.signal_idx[i + 1]) / signal_end, color=curcolor,
                           label=np.str(preictal_signal.file[i][0][0]))

        # ardindan varsa ictal kisim
        if include_ictal:
            seizure_start = (preictal_signal.signal_idx[-1]+ictal_signal.signal_idx[-2])/fs
            sph_start = (preictal_signal.signal_idx[-1])/fs
            axes_0.axvline(x=seizure_start, color='red', linewidth=1.5, label='Seizure Start')
            axes_0.axvspan(xmin=seizure_start, xmax=signal_end/fs, facecolor=(1, 0.8, 0.8), label='Seizure')
            axes_0.axvspan(xmin=sph_start, xmax=seizure_start, facecolor=(1, 1, 0.7), label='SPH')
            axes_1.axvline(x=seizure_start/window_length, color='red', linewidth=1.5, label='Seizure Start')
            axes_1.axvspan(xmin=seizure_start/window_length, xmax=(signal_end/fs)/window_length,
                           facecolor=(1, 0.8, 0.8), label='Seizure')
            axes_1.axvspan(xmin=sph_start/window_length, xmax=seizure_start/window_length, facecolor=(1, 1, 0.7),
                           label='SPH')

        axes_0.legend(bbox_to_anchor=(0.995, 1), loc=2, fancybox=False, shadow=False)
        fig.show()




        # grafikler olusturuluyor
        # plot interictal
        fig = plt.figure()
        gs = gridspec.GridSpec(3, 1, height_ratios=[4, 2, 1])
        axes_0 = plt.subplot(gs[0])
        axes_2 = plt.subplot(gs[1])
        axes_1 = plt.subplot(gs[2])

        fig.suptitle('Patient {0} - Seizure {1}'.format(dataset.pat_id, seizure+1), fontweight='bold')

        # interictal cizdiriliyor
        signal = interictal_signal.signal
        signal_start = interictal_signal.signal_idx[0]
        signal_end = interictal_signal.signal_idx[-1]

        part_count = len(interictal_signal.signal_idx) - 1
        colors = iter(cm.rainbow(np.linspace(0, 1, part_count)))

        axes_0 = stackplot_my(signal,
                            seconds=(signal_end - signal_start)/fs,
                            start_time=signal_start/fs, ylabels=signal_labels,
                            dataRange=[-1024 * 0.5, 1024 * 0.5], my_figure=axes_0, channels=[])
        axes_0.set_title('Interictal EEG Signal')

        # cikarilan featurelar
        axes_2.set_title('Interictal Features')
        test_interictal_feat = np.concatenate([i for i in test_interictal_feats]).swapaxes(0, 1)
        test_interictal_feat = np.asarray([moving_average(i, 3) for i in test_interictal_feat]).swapaxes(0, 1)
        axes_2.plot(test_interictal_feat)
        axes_2.legend(labels)
        axes_2.set_xlim(0, (signal_end / fs) / window_length)
        axes_2.set_ylim(-3, 3)
        axes_2.set_title('Interictal Features')


        # prediction sonuclari
        interictal_pred_avg = moving_average(interictal_pred, 3)
        axes_1.plot(interictal_pred_avg)
        axes_1.set_xlim(0, (signal_end/fs)/window_length)
        axes_1.set_ylim(-0.1, 1.1)
        axes_1.set_title('Predictions')

        #interictal kisim
        for i in range(len(interictal_signal.signal_idx) - 1):
            curcolor = next(colors)
            axes_0.axhspan(axes_0.get_ylim()[1] - subtitle_len, axes_0.get_ylim()[1],
                           float(interictal_signal.signal_idx[i]) / signal_end,
                           float(interictal_signal.signal_idx[i + 1]) / signal_end, color=curcolor,
                           label=np.str(interictal_signal.file[i][0][0]))
            axes_1.axhspan(axes_1.get_ylim()[1] - 0.05, axes_1.get_ylim()[1],
                           float(interictal_signal.signal_idx[i]) / signal_end,
                           float(interictal_signal.signal_idx[i + 1]) / signal_end, color=curcolor,
                           label=np.str(interictal_signal.file[i][0][0]))

        axes_0.legend(bbox_to_anchor=(0.995, 1), loc=2, fancybox=False, shadow=False)
        fig.show()

        # plot preictal features
        # fig, axes = plt.subplots(2,1)
        # fig.suptitle('Patient {0} - Seizure {1}'.format(dataset.pat_id, seizure), fontweight='bold')
        # test_preictal_feat = test_preictal_feats.swapaxes(0, 1)
        # test_preictal_feat = np.asarray([moving_average(i, 30) for i in test_preictal_feat]).swapaxes(0, 1)
        # axes[0].plot(test_preictal_feat)
        # axes[0].legend(labels)
        # axes[0].set_title('Preictal Features')
        #
        #
        # axes[1].plot(moving_average(preictal_pred, 30))
        # axes[1].set_title('Predictions')
        # axes[1].set_xlim(0, len(preictal_pred)+1)
        #
        # fig.show()
        #
        # # plot interictal features
        # fig, axes = plt.subplots(2,1)
        # fig.suptitle('Patient {0} - Seizure {1}'.format(dataset.pat_id, seizure), fontweight='bold')
        # test_interictal_feat = np.concatenate([i for i in test_interictal_feats]).swapaxes(0, 1)
        # test_interictal_feat = np.asarray([moving_average(i, 30) for i in test_interictal_feat]).swapaxes(0, 1)
        # axes[0].plot(test_interictal_feat)
        # axes[0].legend(labels)
        # axes[0].set_title('Interictal Features')
        #
        #
        # axes[1].plot(moving_average(interictal_pred, 30))
        # axes[1].set_title('Predictions')
        # fig.show()




        # pre_fold_idx = fold_pairs[fold][4][0]
    # int_fold_idx = fold_pairs[fold][4][1]


def plot_only_prediction(predict_int,predict_pre):
    fig, axes = plt.subplots(2, 1)
    fig.suptitle('Prediction Results', fontweight='bold')

    predict_int_avg=moving_average(predict_int, 30)
    axes[0].plot(predict_int_avg)
    axes[0].set_title('Interictal Predictions')
    axes[0].set_xlim(0, len(predict_int_avg))
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=max(predict_int_avg), color='red', linewidth=0.5, label=str(max(predict_int_avg)))
    axes[0].legend()

    predict_pre_avg = moving_average(predict_pre, 30)
    axes[1].plot(predict_pre_avg)
    axes[1].set_title('Precital Predictions')
    axes[1].set_xlim(0, len(predict_pre_avg))
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(y=max(predict_pre_avg), color='red', linewidth=0.5, label=str(max(predict_pre_avg)))
    axes[1].legend()

    fig.show()
