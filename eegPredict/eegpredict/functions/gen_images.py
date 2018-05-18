import scipy.io
import numpy as np
import itertools

from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from eegpredict.functions.utils import augment_EEG, chunks, chunkIt
from eegpredict.functions import globals


def find_preictal_interictal_pair(pre_idx, feature_info=None):
    # preictal ve ondan onceki ilk interictal eslestiriliyor
    # oncer interictal yoksa sonraki ilk interictal ile eslestiriliyor
    recordList = np.squeeze(scipy.io.loadmat(globals.records_file)['records'])
    recordListFiles = [recordList[i][0][0][0][0] for i in range(len(recordList))]
    int_feats = feature_info[4][0][0]
    int_feats_file = int_feats[1]
    pre_feats = feature_info[5][0][0]
    pre_feats_file = pre_feats[1][list(np.where(pre_idx + 1 == pre_feats[0])[0])]
    pre_int_pair = []
    for f in pre_feats_file:
        if f in int_feats_file:
            int_pair_idx = np.where(np.array(int_feats_file) == f[0])[0][0]
        else:
            record_list_idx = np.where(np.array(recordListFiles) == f[0])[0][0]
            for i in range(record_list_idx, 0, -1):
                if recordListFiles[i] in int_feats_file:
                    int_pair_idx = np.where(np.array(int_feats_file) == recordListFiles[i])[0][0]
                    break
            for i in range(record_list_idx, len(recordListFiles), 1):
                if recordListFiles[i] in int_feats_file:
                    int_pair_idx = np.where(np.array(int_feats_file) == recordListFiles[i])[0][0]
                    break
        pre_int_pair.append(int_pair_idx)
    return pre_int_pair


def gen_images_basic(locs, features, dims=[4, 5], normalize=False):
    feat_array_temp = []
    nElectrodes = locs.shape[0]  # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] / nElectrodes
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes: nElectrodes * (c + 1)])
    nSamples = features.shape[0]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, dims[0], dims[1]]))
    for c in range(n_colors):
        for i in range(locs.shape[0]):
            temp_interp[c][:, locs[i][0], locs[i][1]] = feat_array_temp[c][:, i]
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])], with_mean=True)
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)


def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]  # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] / nElectrodes
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes: nElectrodes * (c + 1)])
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints * 1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints * 1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    # Interpolating
    for i in xrange(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                               method='cubic', fill_value=np.nan)
            print('Interpolating {0}/{1}\r'.format(i+1, nSamples))
    print('Interpolating...\r')
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])], with_mean=False)
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)  # swap axes to have [samples, colors, W, H]


def gen_image_dataset(locs_2d, pat_id, feature_info, feature_size=14 * 20, pixelCount=[8, 8], gen_image='basic'):
    # Preictal Goruntu Olusturma
    feature_pre = scipy.io.loadmat('../../generateEEGFeats/dataset/pat_' + str(pat_id) + '_pre.mat')['feature']
    feature_pre = np.asarray(feature_pre)
    wCount_pre = feature_pre.shape[0]

    # Interictal Goruntu Olusturma
    feature_int = scipy.io.loadmat('../../generateEEGFeats/dataset/pat_' + str(pat_id) + '_int.mat')['feature']
    feature_int = np.asarray(feature_int)
    wCount_int = feature_int.shape[0]

    feature = np.concatenate((feature_pre[:, 0:feature_size], feature_int[:, 0:feature_size]), axis=0)

    if gen_image == 'basic':
        images_timewin = gen_images_basic(np.array(locs_2d), feature, normalize=False)
    elif gen_image == 'advanced':
        images_timewin = gen_images(np.array(locs_2d), feature, pixelCount[0], normalize=False,
                                    edgeless=False)


    scipy.io.savemat('../data/pat_' + str(pat_id) + '_pre_' + str(pixelCount[1]) + 'px.mat',
                     {'images_timewin_pre': images_timewin[0:wCount_pre]})

    scipy.io.savemat('../data/pat_' + str(pat_id) + '_int_' + str(pixelCount[1]) + 'px.mat',
                     {'images_timewin_int': images_timewin[wCount_pre:]})


    # Etiketler de kaydediliyor
    pat_pre_lbl = feature_pre[:, -2:].astype(np.int32)
    scipy.io.savemat('../data/pat_' + str(pat_id) + '_pre_label.mat',
                     {'pat_pre_lbl': pat_pre_lbl})
    pat_int_lbl = feature_int[:, -1:].astype(np.int32)
    scipy.io.savemat('../data/pat_' + str(pat_id) + '_int_label.mat',
                     {'pat_int_lbl': pat_int_lbl})


def organize_dataset(images_timewin_pos, images_timewin_neg, seq_win_count=5, s_count=[5, 5]):
    # pozitif orneklerden rastgele pencere dizisi secme
    # 5 farkli pencere secilecek
    s_count_pos = s_count[0]
    images_timewin_p = []
    for idx in range(images_timewin_pos.shape[1]):
        w_count = images_timewin_pos.shape[0]
        w_index = range(w_count - (seq_win_count - 1))
        np.random.shuffle(w_index)
        for sc in range(s_count_pos):
            images_timewin_p.append(images_timewin_pos[w_index[sc]:w_index[sc] + seq_win_count, idx])
    images_timewin_p = np.swapaxes(np.asarray(images_timewin_p), 0, 1)

    # negatif orneklerden rastgele pencere dizisi secme
    # 5 farkli pencere secilecek
    s_count_neg = s_count[1]
    images_timewin_n = []
    for idx in range(images_timewin_neg.shape[1]):
        w_count = images_timewin_neg.shape[0] / seq_win_count
        w_index = range(w_count)
        np.random.shuffle(w_index)
        for sc in range(s_count_neg):
            images_timewin_n.append(
                images_timewin_neg[w_index[sc] * seq_win_count:(w_index[sc] + 1) * seq_win_count, idx])
    images_timewin_n = np.swapaxes(np.asarray(images_timewin_n), 0, 1)

    # pozitif test indexleri
    # train, validation ve test verilerinin icice gecmesi engelleniyor
    index_pos = np.arange(0, images_timewin_p.shape[1], s_count_pos)
    np.random.shuffle(index_pos)
    ts = np.asarray(index_pos[:(images_timewin_pos.shape[1] / 5)])
    index_pos = np.setdiff1d(index_pos, ts)
    # pozitif valid indexleri
    np.random.shuffle(index_pos)
    vl = index_pos[:(images_timewin_pos.shape[1] / 5)]
    # pozitif train indexleri
    tr = np.setdiff1d(index_pos, vl)

    tr_pos = tr
    vl_pos = vl
    ts_pos = ts
    for i in np.arange(1, s_count_pos):
        tr_pos = np.concatenate((tr_pos, tr + i))
        vl_pos = np.concatenate((vl_pos, vl + i))
        ts_pos = np.concatenate((ts_pos, ts + i))

    # negatif test indexleri
    index_neg = np.arange(0, images_timewin_n.shape[1])
    np.random.shuffle(index_neg)
    # negatif test sayisi pozitif ile ayni
    ts_neg = np.asarray(index_neg[:(ts_pos.shape[0])])
    index_neg = np.setdiff1d(index_neg, ts_neg)
    # negatif valid indexleri
    np.random.shuffle(index_neg)
    # negatif valid sayisi pozitif ile ayni
    vl_neg = index_neg[:(vl_pos.shape[0])]
    # negatif train indexleri
    tr_neg = np.setdiff1d(index_neg, vl_neg)

    # pozitif ve negatif indeksler kaskad eklenecek
    tr_neg += images_timewin_p.shape[1]
    vl_neg += images_timewin_p.shape[1]
    ts_neg += images_timewin_p.shape[1]

    tr = np.concatenate((tr_pos, tr_neg))
    vl = np.concatenate((vl_pos, vl_neg))
    ts = np.concatenate((ts_pos, ts_neg))

    # train, valid ve test indeksleri shuffle ediliyor
    np.random.shuffle(tr)
    np.random.shuffle(vl)
    np.random.shuffle(ts)

    fold_pairs = []
    fold_pairs.append((tr, vl, ts))

    # labels
    n_labels = np.full([images_timewin_n.shape[1]], 0)
    p_labels = np.full([images_timewin_p.shape[1]], 1)
    labels = np.concatenate((p_labels, n_labels))
    images_timewin = np.concatenate((images_timewin_p, images_timewin_n), axis=1)

    ratio = tr_neg.shape[0] / float(tr_pos.shape[0])

    return images_timewin, labels, fold_pairs[0], ratio


def organize_dataset_imbalanced(pre_label, int_label, seq_win_count=5, fold_count=5):
    # kac adet preictal kayit varsa interictal o kadar parcaya ayrilacak
    # 1 preictal ve interictal parca test icin ayrilacak (out-of-sample testing) digerleri 5'li capraz dogrulamaya tabi tutulacak ( in-sample optimization)
    pre_u, pre_idx = np.unique(pre_label[:, 1], return_index=True)
    pre_idx = np.append(pre_idx, pre_label.shape[0])
    int_u, int_idx = np.unique(int_label, return_index=True)
    int_idx = np.append(int_idx, int_label.shape[0])
    # interictal parcalar rastgele eslestirilecek
    int_p_size = int_label.shape[0] / pre_u.shape[0]
    int_p_idx = range(0, int_label.shape[0]+1, int_p_size)
    int_p_idx[-1] = int_label.shape[0]
    int_fold_idx = range(pre_u.shape[0])
    np.random.shuffle(int_fold_idx)

    # tum preictal ve interictal indexleri
    # kayit sonlarindaki seq_win_count kadar veriyi cikarmak icin asagidaki kodlari ekledik
    all_pre = []
    for i in range(pre_u.shape[0]):
        p_idx = range(pre_idx[i], pre_idx[i + 1])
        u, idx = np.unique(pre_label[p_idx, 0], return_index=True)
        idx = np.append(idx, np.asarray(p_idx).shape[0])
        part_pre = list(itertools.chain.from_iterable(
            [range(p_idx[idx[j]],  p_idx[idx[j + 1]-1]+1 - (seq_win_count-1)) for j in range(idx.shape[0] - 1)]))
        all_pre.append(part_pre)
    all_pre = list(itertools.chain.from_iterable(all_pre))
    all_int = list(itertools.chain.from_iterable(
        [range(int_idx[i], int_idx[i + 1] - (seq_win_count-1)) for i in range(int_u.shape[0])]))
    ratio = np.asarray(all_int).shape[0] / float(np.asarray(all_pre).shape[0])

    fold_pairs = []

    # seizure sayisi kadar fold pair olacak
    for i in range(pre_u.shape[0]):
        # test indexleri ayriliyor
        test_pre = range(pre_idx[i], pre_idx[i + 1])
        test_int = range(int_p_idx[int_fold_idx[i]], int_p_idx[int_fold_idx[i] + 1])

        # train indexleri
        train_pre = np.setdiff1d(all_pre, test_pre)
        train_int = np.setdiff1d(all_int, test_int)

        # test indexlerinden dosya sonundaki seq_win_count-1 index cikariliyor
        test_pre = np.setdiff1d(all_pre, train_pre)
        test_int = np.setdiff1d(all_int, train_int)

        # test verileri sirali train verileri rastgele olacak
        # ayrica train verileri fold count kadar parcaya ayrilacak
        np.random.shuffle(train_pre)
        np.random.shuffle(train_int)
        train_pre_pairs = np.asarray(list(chunks(train_pre, (train_pre.shape[0] / fold_count) + 1)))
        train_int_pairs = np.asarray(list(chunks(train_int, (train_int.shape[0] / fold_count) + 1)))

        test_idx = [[pre_idx[i], pre_idx[i + 1]], [int_p_idx[int_fold_idx[i]], int_p_idx[int_fold_idx[i] + 1]]]

        fold_pairs.append((train_pre_pairs, train_int_pairs, test_pre, test_int, test_idx))

    return fold_pairs, ratio

def organize_dataset_imbalanced_rand(pre_label, int_label, seq_win_count=5, fold_count=5, feature_info=None):
    # kac adet preictal kayit varsa interictal o kadar parcaya ayrilacak
    # 1 preictal ve interictal parca test icin ayrilacak (out-of-sample testing) digerleri 5'li capraz dogrulamaya tabi tutulacak ( in-sample optimization)
    pre_u, pre_idx = np.unique(pre_label[:, 1], return_index=True)
    pre_idx = np.append(pre_idx, pre_label.shape[0])
    int_u, int_idx = np.unique(int_label, return_index=True)
    int_idx = np.append(int_idx, int_label.shape[0])
    # idx'ler 1 den basliyordu 0 yapildi
    pre_u = pre_u - 1
    int_u = int_u -1

    # preictal ve ondan onceki ilk interictal eslestiriliyor
    recordList = np.squeeze(scipy.io.loadmat(globals.records_file)['records'])
    recordListFiles = [recordList[i][0][0][0][0] for i in range(len(recordList))]
    int_feats = feature_info[4][0][0]
    int_feats_file = int_feats[1]
    pre_feats = feature_info[5][0][0]
    pre_feats_file = pre_feats[1][list(np.where(pre_idx + 1 == pre_feats[0])[0])]
    pre_int_pair = []
    for f in pre_feats_file:
        if f in int_feats_file:
            int_pair_idx = np.where(np.array(int_feats_file) == f[0])[0][0]
        else:
            record_list_idx = np.where(np.array(recordListFiles) == f[0])[0][0]
            for i in range(record_list_idx,0,-1):
                if recordListFiles[i] in int_feats_file:
                    int_pair_idx = np.where(np.array(int_feats_file) == recordListFiles[i])[0][0]
                    break
        pre_int_pair.append(int_pair_idx)



    # interictal parcalar rastgele eslestirilecek
    int_u_diff = np.setdiff1d(int_u, pre_int_pair)
    np.random.shuffle(int_u_diff)
    int_fold_idx = np.array_split(int_u_diff, pre_u.shape[0])

    # interictal parcalar rastgele eslestirilecek
    # np.random.shuffle(int_u)
    # int_fold_idx=np.array_split(int_u, pre_u.shape[0])


    # tum preictal ve interictal indexleri
    all_pre = []
    for i in range(pre_u.shape[0]):
        p_idx = range(pre_idx[i], pre_idx[i + 1])
        u, idx = np.unique(pre_label[p_idx, 0], return_index=True)
        idx = np.append(idx, np.asarray(p_idx).shape[0])
        part_pre = list(itertools.chain.from_iterable(
            [range(p_idx[idx[j]], p_idx[idx[j + 1]-1]+1 - (seq_win_count-1)) for j in range(idx.shape[0] - 1)]))
        all_pre.append(part_pre)
    all_pre = list(itertools.chain.from_iterable(all_pre))
    all_int = list(itertools.chain.from_iterable(
        [range(int_idx[i], int_idx[i + 1] - (seq_win_count-1)) for i in range(int_u.shape[0])]))
    ratio = np.asarray(all_int).shape[0] / float(np.asarray(all_pre).shape[0])

    fold_pairs = []

    # seizure sayisi kadar fold pair olacak
    for i in range(pre_u.shape[0]):
        # test indexleri ayriliyor
        test_pre = range(pre_idx[i], pre_idx[i + 1])
        test_int=[]
        test_int = np.append(test_int, range(int_idx[pre_int_pair[i]], int_idx[pre_int_pair[i] + 1]))
        test_int_idx=[0]
        for k in range(int_fold_idx[i].shape[0]):
            test_int=np.append(test_int, range(int_idx[int_fold_idx[i][k]], int_idx[int_fold_idx[i][k]+1]))
            test_int_idx.append(len(test_int))


        # train indexleri
        train_pre = np.setdiff1d(all_pre, test_pre)
        train_int = np.setdiff1d(all_int, test_int)

        # test indexlerinden dosya sonundaki seq_win_count-1 index cikariliyor
        test_pre = np.setdiff1d(all_pre, train_pre)
        test_int = np.setdiff1d(all_int, train_int)

        # test verileri sirali train verileri rastgele olacak
        # ayrica train verileri fold count kadar parcaya ayrilacak
        np.random.shuffle(train_pre)
        np.random.shuffle(train_int)
        train_pre_pairs = np.asarray(list(chunks(train_pre, (train_pre.shape[0] / fold_count) + 1)))
        train_int_pairs = np.asarray(list(chunks(train_int, (train_int.shape[0] / fold_count) + 1)))

        # pat_info icindeki indeksleri
        test_idx = [i, int_fold_idx[i], test_int_idx]


        fold_pairs.append((train_pre_pairs, train_int_pairs, test_pre, test_int, test_idx))

    return fold_pairs, ratio


def organize_dataset_imbalanced_fold_inpart(pre_label, int_label, seq_win_count=5, fold_count=5, random_fold=True, feature_info=None):
    # kac adet preictal kayit varsa interictal o kadar parcaya ayrilacak
    # 1 preictal ve interictal parca test icin ayrilacak (out-of-sample testing) digerleri 5'li capraz dogrulamaya tabi tutulacak ( in-sample optimization)
    pre_u, pre_idx = np.unique(pre_label[:, 1], return_index=True)
    pre_idx = np.append(pre_idx, pre_label.shape[0])
    int_u, int_idx = np.unique(int_label, return_index=True)
    int_idx = np.append(int_idx, int_label.shape[0])
    # idx'ler 1 den basliyordu 0 yapildi
    pre_u = pre_u - 1
    int_u = int_u -1

    # preictal ve ondan onceki ilk interictal eslestiriliyor
    recordList = np.squeeze(scipy.io.loadmat(globals.records_file)['records'])
    recordListFiles = [recordList[i][0][0][0][0] for i in range(len(recordList))]
    int_feats = feature_info[4][0][0]
    int_feats_file = int_feats[1]
    pre_feats = feature_info[5][0][0]
    pre_feats_file = pre_feats[1][list(np.where(pre_idx + 1 == pre_feats[0])[0])]
    pre_int_pair = []
    for f in pre_feats_file:
        if f in int_feats_file:
            int_pair_idx = np.where(np.array(int_feats_file) == f[0])[0][0]
        else:
            record_list_idx = np.where(np.array(recordListFiles) == f[0])[0][0]
            for i in range(record_list_idx,0,-1):
                if recordListFiles[i] in int_feats_file:
                    int_pair_idx = np.where(np.array(int_feats_file) == recordListFiles[i])[0][0]
                    break
        pre_int_pair.append(int_pair_idx)



    # interictal parcalar rastgele eslestirilecek
    int_u_diff = np.setdiff1d(int_u, pre_int_pair)
    np.random.shuffle(int_u_diff)
    int_fold_idx = np.array_split(int_u_diff, pre_u.shape[0])

    # tum preictal ve interictal indexleri
    all_pre = []
    for i in range(pre_u.shape[0]):
        p_idx = range(pre_idx[i], pre_idx[i + 1])
        u, idx = np.unique(pre_label[p_idx, 0], return_index=True)
        idx = np.append(idx, np.asarray(p_idx).shape[0])
        part_pre = list(itertools.chain.from_iterable(
            [range(p_idx[idx[j]], p_idx[idx[j + 1]-1]+1 - (seq_win_count-1)) for j in range(idx.shape[0] - 1)]))
        all_pre.append(part_pre)
    all_int = [range(int_idx[i], int_idx[i + 1] - (seq_win_count-1)) for i in range(int_u.shape[0])]

    #ratio = np.asarray(all_int).shape[0] / float(np.asarray(all_pre).shape[0])

    fold_pairs = []

    # seizure sayisi kadar fold pair olacak
    for i in range(pre_u.shape[0]):
        # ic int ve pre listeleri olusturuluyor

        # test indexleri ayriliyor
        test_pre = all_pre[i]
        test_int = []
        test_int = np.append(test_int, all_int[pre_int_pair[i]])
        pre_int_pair_diff = np.setdiff1d(pre_int_pair, pre_int_pair[i])
        test_int_idx=[0]
        test_int_len=0
        for k in range(int_fold_idx[i].shape[0]):
            test_int = np.append(test_int, all_int[int_fold_idx[i][k]])
            test_int_len += len(range(int_idx[int_fold_idx[i][k]], int_idx[int_fold_idx[i][k]+1]))
            test_int_idx.append(test_int_len)

        train_int_fold_idx = np.setdiff1d(list(itertools.chain.from_iterable([j for j in int_fold_idx])), int_fold_idx[i])
        train_int_fold_idx = np.append(train_int_fold_idx, pre_int_pair_diff)
        train_pre_fold_idx = np.setdiff1d(range(len(pre_u)), i)
        # train indexleri
        train_pre = [all_pre[j] for j in train_pre_fold_idx]
        train_int = [all_int[j] for j in train_int_fold_idx]

        fold_range=range(fold_count)

        train_pre_pairs = [list(chunks(j, (len(j) / fold_count) + 1)) for j in train_pre]
        if random_fold:
            for a, b in zip(train_pre_pairs, range(len(train_pre_pairs))):
                for bb in range(b):
                    np.random.shuffle(a)
        train_pre_pairs = np.transpose(train_pre_pairs)
        train_pre_pairs = np.asarray([np.hstack(j) for j in train_pre_pairs])

        train_int_pairs = [list(chunks(j, (len(j) / fold_count) + 1)) for j in train_int]
        if random_fold:
            for a, b in zip(train_int_pairs, range(len(train_int_pairs))):
                for bb in range(b):
                    np.random.shuffle(a)
        train_int_pairs = np.transpose(train_int_pairs)
        train_int_pairs = np.asarray([np.hstack(j) for j in train_int_pairs])

        # pat_info icindeki indeksleri
        test_idx = [i, int_fold_idx[i], test_int_idx]

        ratio=0


        fold_pairs.append((train_pre_pairs, train_int_pairs, np.asarray(test_pre).astype(int), np.asarray(test_int).astype(int), test_idx))

    return fold_pairs, ratio


def organize_dataset_imbalanced_independent(pre_label, int_label, seq_win_count=5, fold_count=5, random_fold=True, feature_info=None):
    # kac adet preictal kayit varsa interictal o kadar parcaya ayrilacak
    # 1 preictal ve interictal parca test icin ayrilacak (out-of-sample testing) digerleri 5'li capraz dogrulamaya tabi tutulacak ( in-sample optimization)
    pre_u, pre_idx = np.unique(pre_label[:, 1], return_index=True)
    pre_idx = np.append(pre_idx, pre_label.shape[0])
    int_u, int_idx = np.unique(int_label, return_index=True)
    int_idx = np.append(int_idx, int_label.shape[0])
    # idx'ler 1 den basliyordu 0 yapildi
    pre_u = pre_u - 1
    int_u = int_u -1

    np.random.shuffle(int_u)
    int_fold_idx = np.array_split(int_u, pre_u.shape[0])

    # tum preictal ve interictal indexleri
    all_pre = []
    for i in range(pre_u.shape[0]):
        p_idx = range(pre_idx[i], pre_idx[i + 1])
        u, idx = np.unique(pre_label[p_idx, 0], return_index=True)
        idx = np.append(idx, np.asarray(p_idx).shape[0])
        part_pre = list(itertools.chain.from_iterable(
            [range(p_idx[idx[j]], p_idx[idx[j + 1]-1]+1 - (seq_win_count-1)) for j in range(idx.shape[0] - 1)]))
        all_pre.append(part_pre)
    all_int = [range(int_idx[i], int_idx[i + 1] - (seq_win_count-1)) for i in range(int_u.shape[0])]

    #ratio = np.asarray(all_int).shape[0] / float(np.asarray(all_pre).shape[0])

    fold_pairs = []

    # seizure sayisi kadar fold pair olacak
    for i in range(pre_u.shape[0]):
        # ic int ve pre listeleri olusturuluyor

        # test indexleri ayriliyor
        test_pre = all_pre[i]
        test_int = []
        test_int_idx = [0]
        test_int_len = 0
        test_int_idx.append(test_int_len)
        for k in range(int_fold_idx[i].shape[0]):
            test_int = np.append(test_int, all_int[int_fold_idx[i][k]])
            test_int_len += len(range(int_idx[int_fold_idx[i][k]], int_idx[int_fold_idx[i][k]+1]))
            test_int_idx.append(test_int_len)

        train_int_fold_idx = np.setdiff1d(list(itertools.chain.from_iterable([j for j in int_fold_idx])), int_fold_idx[i])
        train_pre_fold_idx = np.setdiff1d(range(len(pre_u)), i)
        # train indexleri
        train_pre = [all_pre[j] for j in train_pre_fold_idx]
        train_int = [all_int[j] for j in train_int_fold_idx]

        tr_pre_pairs = [list(chunks(j, (len(j) / fold_count) + 1)) for j in train_pre]
        tr_pre_pairs = np.hstack(tr_pre_pairs)
        np.random.shuffle(tr_pre_pairs)

        valid_pre_part_count = np.int(np.ceil(np.float(len(tr_pre_pairs)) / fold_count))
        # valid_pre_part_count = np.int(np.ceil(np.float(len(train_pre))/fold_count))
        valid_int_part_count = np.int(np.ceil(np.float(len(train_int)) / fold_count))

        np.random.shuffle(train_pre)
        np.random.shuffle(train_int)

        train_pre_pairs = []
        train_int_pairs = []
        for n in range(fold_count):
            # valid_pre_idx = np.random.randint(0, len(train_pre), valid_pre_part_count)
            # valid_pre_pair = np.hstack([train_pre[j] for j in valid_pre_idx])
            # valid_int_idx = np.random.randint(0, len(train_int), valid_int_part_count)
            # valid_int_pair = np.hstack([train_int[j] for j in valid_int_idx])
            valid_pre_pair = np.hstack(tr_pre_pairs[n:n+valid_pre_part_count])
            valid_int_pair = np.hstack(train_int[n:n + valid_int_part_count])

            train_pre_pair = np.setdiff1d(np.hstack(train_pre), valid_pre_pair)
            train_int_pair = np.setdiff1d(np.hstack(train_int), valid_int_pair)

            train_pre_pairs.append([train_pre_pair, valid_pre_pair])
            train_int_pairs.append([train_int_pair, valid_int_pair])

        # pat_info icindeki indeksleri
        test_idx = [i, int_fold_idx[i], test_int_idx]

        ratio = 0

        fold_pairs.append((train_pre_pairs, train_int_pairs, np.asarray(test_pre).astype(int), np.asarray(test_int).astype(int), test_idx))

    return fold_pairs, ratio


def organize_dataset_imbalanced_last(pre_label, int_label, seq_win_count=5, fold_count=5, random_fold=True, feature_info=None):
    # kac adet preictal kayit varsa interictal o kadar parcaya ayrilacak
    # 1 preictal ve interictal parca test icin ayrilacak (out-of-sample testing) digerleri 5'li capraz dogrulamaya tabi tutulacak ( in-sample optimization)
    pre_u, pre_idx = np.unique(pre_label[:, 1], return_index=True)
    pre_idx = np.append(pre_idx, pre_label.shape[0])
    int_u, int_idx = np.unique(int_label, return_index=True)
    int_idx = np.append(int_idx, int_label.shape[0])
    # idx'ler 1 den basliyordu 0 yapildi
    pre_u = pre_u - 1
    int_u = int_u -1

    np.random.shuffle(int_u)
    int_fold_idx = np.array_split(int_u, pre_u.shape[0])

    # tum preictal ve interictal indexleri
    all_pre = []
    for i in range(pre_u.shape[0]):
        p_idx = range(pre_idx[i], pre_idx[i + 1])
        u, idx = np.unique(pre_label[p_idx, 0], return_index=True)
        idx = np.append(idx, np.asarray(p_idx).shape[0])
        part_pre = list(itertools.chain.from_iterable(
            [range(p_idx[idx[j]], p_idx[idx[j + 1]-1]+1 - (seq_win_count-1)) for j in range(idx.shape[0] - 1)]))
        all_pre.append(part_pre)
    all_int = [range(int_idx[i], int_idx[i + 1] - (seq_win_count-1)) for i in range(int_u.shape[0])]

    #ratio = np.asarray(all_int).shape[0] / float(np.asarray(all_pre).shape[0])

    fold_pairs = []

    # seizure sayisi kadar fold pair olacak
    for i in range(pre_u.shape[0]):
        # ic int ve pre listeleri olusturuluyor

        # test indexleri ayriliyor
        test_pre = all_pre[i]
        test_int = []
        test_int_idx = [0]
        test_int_len = 0
        test_int_idx.append(test_int_len)
        for k in range(int_fold_idx[i].shape[0]):
            test_int = np.append(test_int, all_int[int_fold_idx[i][k]])
            test_int_len += len(range(int_idx[int_fold_idx[i][k]], int_idx[int_fold_idx[i][k]+1]))
            test_int_idx.append(test_int_len)

        train_int_fold_idx = np.setdiff1d(list(itertools.chain.from_iterable([j for j in int_fold_idx])), int_fold_idx[i])
        train_pre_fold_idx = np.setdiff1d(range(len(pre_u)), i)
        # train indexleri
        train_pre = [all_pre[j] for j in train_pre_fold_idx]
        train_int = [all_int[j] for j in train_int_fold_idx]

        tr_pre_pairs = [list(chunks(j, (len(j) / fold_count) + 1)) for j in train_pre]
        tr_pre_pairs_valid = np.hstack(tr_pre_pairs[j][-1] for j in range(len(tr_pre_pairs)))

        tr_int_pairs = [list(chunks(j, (len(j) / fold_count) + 1)) for j in train_int]
        tr_int_pairs_valid = np.hstack(tr_int_pairs[j][-1] for j in range(len(tr_int_pairs)))

        # valid_pre_part_count = np.int(np.ceil(np.float(len(tr_pre_pairs)) / fold_count))
        # valid_int_part_count = np.int(np.ceil(np.float(len(tr_int_pairs)) / fold_count))
        # valid_pre_part_count = np.int(np.ceil(np.float(len(train_pre))/fold_count))
        # valid_int_part_count = np.int(np.ceil(np.float(len(train_int)) / fold_count))

        np.random.shuffle(train_pre)
        np.random.shuffle(train_int)

        train_pre_pairs = []
        train_int_pairs = []
        for n in range(fold_count):
            # valid_pre_idx = np.random.randint(0, len(train_pre), valid_pre_part_count)
            # valid_pre_pair = np.hstack([train_pre[j] for j in valid_pre_idx])
            # valid_int_idx = np.random.randint(0, len(train_int), valid_int_part_count)
            # valid_int_pair = np.hstack([train_int[j] for j in valid_int_idx])
            # valid_pre_pair = np.hstack(tr_pre_pairs[n:n+valid_pre_part_count])
            valid_pre_pair = tr_pre_pairs_valid
            valid_int_pair = tr_int_pairs_valid
            # valid_int_pair = np.hstack(train_int[n:n + valid_int_part_count])

            train_pre_pair = np.setdiff1d(np.hstack(train_pre), valid_pre_pair)
            train_int_pair = np.setdiff1d(np.hstack(train_int), valid_int_pair)

            train_pre_pairs.append([train_pre_pair, valid_pre_pair])
            train_int_pairs.append([train_int_pair, valid_int_pair])

        # pat_info icindeki indeksleri
        test_idx = [i, int_fold_idx[i], test_int_idx]

        ratio = 0

        fold_pairs.append((train_pre_pairs, train_int_pairs, np.asarray(test_pre).astype(int), np.asarray(test_int).astype(int), test_idx))

    return fold_pairs, ratio


def organize_dataset_imbalanced_conf(pre_label, int_label, seq_win_count=5, fold_count=5, feature_info=None,
                                     preictal_with_int_pair=True, train_valid_method='all_random'):
    # kac adet preictal kayit varsa interictal o kadar parcaya ayrilacak
    # 1 preictal ve interictal parca test icin ayrilacak (out-of-sample testing) digerleri 5'li capraz dogrulamaya tabi tutulacak ( in-sample optimization)
    pre_u, pre_idx = np.unique(pre_label[:, 1], return_index=True)
    pre_idx = np.append(pre_idx, pre_label.shape[0])
    int_u, int_idx = np.unique(int_label, return_index=True)
    int_idx = np.append(int_idx, int_label.shape[0])
    # idx'ler 1 den basliyordu 0 yapildi
    pre_u = pre_u - 1
    int_u = int_u -1


    # kayit sonlarindaki seq_win_count kadar veriyi cikarmak icin asagidaki kodlari ekledik
    all_pre = []
    for i in range(pre_u.shape[0]):
        p_idx = range(pre_idx[i], pre_idx[i + 1])
        u, idx = np.unique(pre_label[p_idx, 0], return_index=True)
        idx = np.append(idx, np.asarray(p_idx).shape[0])
        part_pre = list(itertools.chain.from_iterable(
            [range(p_idx[idx[j]], p_idx[idx[j + 1]-1]+1 - (seq_win_count-1)) for j in range(idx.shape[0] - 1)]))
        all_pre.append(part_pre)
    all_int = [range(int_idx[i], int_idx[i + 1] - (seq_win_count-1)) for i in range(int_u.shape[0])]


    # preictal ve ondan onceki ilk interictal eslestiriliyor
    if preictal_with_int_pair:
        pre_int_pair = find_preictal_interictal_pair(pre_idx, feature_info=feature_info)
        int_u = np.setdiff1d(int_u, pre_int_pair)

    np.random.shuffle(int_u)
    int_fold_idx = np.array_split(int_u, pre_u.shape[0])

    fold_pairs = []

    # seizure sayisi kadar fold pair olacak
    for i in range(pre_u.shape[0]):
        # test indexleri ayriliyor
        test_pre = all_pre[i]
        test_int = []
        test_int_idx = []
        test_int_len = 0
        test_int_idx.append(test_int_len)
        # once diger interictal test parcalari
        for k in range(int_fold_idx[i].shape[0]):
            test_int = np.append(test_int, all_int[int_fold_idx[i][k]])
            test_int_len += len(range(int_idx[int_fold_idx[i][k]], int_idx[int_fold_idx[i][k]+1]))
            test_int_idx.append(test_int_len)
        # preictal oncesi(ya da sonrasi) interictal parca
        if preictal_with_int_pair:
            test_int = np.append(test_int, all_int[pre_int_pair[i]])
            test_int_len += len(range(int_idx[pre_int_pair[i]], int_idx[pre_int_pair[i] + 1]))
            test_int_idx.append(test_int_len)
            pre_int_pair_diff = np.setdiff1d(pre_int_pair, pre_int_pair[i])

        # tum interictal ve preictal parcalardan test icin ayrilanlari cikariliyor ve train verileri elde ediliyor
        train_int_fold_idx = np.setdiff1d(list(itertools.chain.from_iterable([j for j in int_fold_idx])), int_fold_idx[i])
        train_pre_fold_idx = np.setdiff1d(range(len(pre_u)), i)
        # preictal oncesi(ya da sonrasi) interictal parca
        if preictal_with_int_pair:
            train_int_fold_idx = np.append(train_int_fold_idx, pre_int_pair_diff)

        # train indexleri
        train_pre = [all_pre[j] for j in train_pre_fold_idx]
        train_int = [all_int[j] for j in train_int_fold_idx]

        train_pre_pairs = []
        train_int_pairs = []

        if train_valid_method == 'all_random':
            # tum preictal ve interictal veriler birlestirilip karistiriliyor
            train_pre = np.hstack(train_pre)
            train_int = np.hstack(train_int)
            np.random.shuffle(train_pre)
            np.random.shuffle(train_int)

            # fold kadar parcaya bolunuyor
            tr_pre_pairs = [c for c in chunks(train_pre, (len(train_pre) / fold_count) + 1)]
            tr_int_pairs = [c for c in chunks(train_int, (len(train_int) / fold_count) + 1)]

            for n in range(fold_count):
                valid_pre_pair = tr_pre_pairs[n]
                valid_int_pair = tr_int_pairs[n]

                train_pre_pair = np.setdiff1d(np.hstack(train_pre), valid_pre_pair)
                train_int_pair = np.setdiff1d(np.hstack(train_int), valid_int_pair)

                np.random.shuffle(train_pre_pair)
                np.random.shuffle(train_int_pair)

                train_pre_pairs.append([train_pre_pair, valid_pre_pair])
                train_int_pairs.append([train_int_pair, valid_int_pair])

        if train_valid_method == 'partial_fold':
            if len(train_pre) < fold_count:
                valid_pre_idx = np.random.randint(0, len(train_pre), fold_count)
            else:
                part_pre_idx = range(len(train_pre))
                np.random.shuffle(part_pre_idx)
                valid_pre_idx = chunkIt(part_pre_idx, fold_count)
            if len(train_int) < fold_count:
                valid_int_idx = np.random.randint(0, len(train_int), fold_count)
            else:
                part_int_idx = range(len(train_int))
                np.random.shuffle(part_int_idx)
                valid_int_idx = chunkIt(part_int_idx, fold_count)

            for n in range(fold_count):
                if np.asarray(valid_pre_idx[n]).ndim:
                    valid_pre_pair = np.hstack([train_pre[j] for j in valid_pre_idx[n]])
                else:
                    valid_pre_pair = np.asarray(train_pre[valid_pre_idx[n]])
                if np.asarray(valid_int_idx[n]).ndim:
                    valid_int_pair = np.hstack([train_int[j] for j in valid_int_idx[n]])
                else:
                    valid_int_pair = np.asarray(train_int[valid_int_idx[n]])
                np.random.shuffle(valid_pre_pair)
                np.random.shuffle(valid_int_pair)

                train_pre_pair = np.setdiff1d(np.hstack(train_pre), valid_pre_pair)
                train_int_pair = np.setdiff1d(np.hstack(train_int), valid_int_pair)

                np.random.shuffle(train_pre_pair)
                np.random.shuffle(train_int_pair)

                train_pre_pairs.append([train_pre_pair, valid_pre_pair])
                train_int_pairs.append([train_int_pair, valid_int_pair])
        if train_valid_method == 'split_fold':
            # preictal ve interictal parcalar kendi icinde fold kadar parcaya ayriliyor
            # ve bu parcalar rastgele hale getiriliyor
            tr_pre_pairs = [list(chunkIt(j, fold_count)) for j in train_pre]
            [np.random.shuffle(tr_pre_pairs[j]) for j in range(len(tr_pre_pairs))]
            tr_pre_pairs = np.moveaxis(tr_pre_pairs, 0, 1)

            tr_int_pairs = [list(chunkIt(j, fold_count)) for j in train_int]
            [np.random.shuffle(tr_int_pairs[j]) for j in range(len(tr_int_pairs))]
            tr_int_pairs = np.moveaxis(tr_int_pairs, 0, 1)

            for n in range(fold_count):
                valid_pre_pair = np.hstack([j for j in tr_pre_pairs[n]])
                valid_int_pair = np.hstack([j for j in tr_int_pairs[n]])

                train_pre_pair = np.setdiff1d(np.hstack(train_pre), valid_pre_pair)
                train_int_pair = np.setdiff1d(np.hstack(train_int), valid_int_pair)

                np.random.shuffle(train_pre_pair)
                np.random.shuffle(train_int_pair)

                train_pre_pairs.append([train_pre_pair, valid_pre_pair])
                train_int_pairs.append([train_int_pair, valid_int_pair])

        # pat_info icindeki indeksleri
        test_idx = [i, int_fold_idx[i], test_int_idx]

        ratio = 0

        fold_pairs.append((train_pre_pairs, train_int_pairs, np.asarray(test_pre).astype(int), np.asarray(test_int).astype(int), test_idx, pre_idx))

    return fold_pairs, ratio


def organize_dataset_pretrain(pre_label, int_label, seq_win_count=5, under_sample_ratio=0.1):
    # kac adet preictal kayit varsa interictal o kadar parcaya ayrilacak
    # 1 preictal ve interictal parca test icin ayrilacak (out-of-sample testing) digerleri 5'li capraz dogrulamaya tabi tutulacak ( in-sample optimization)
    pre_u, pre_idx = np.unique(pre_label[:, 1], return_index=True)
    pre_idx = np.append(pre_idx, pre_label.shape[0])
    int_u, int_idx = np.unique(int_label, return_index=True)
    int_idx = np.append(int_idx, int_label.shape[0])
    # idx'ler 1 den basliyordu 0 yapildi
    pre_u = pre_u - 1
    int_u = int_u -1


    # kayit sonlarindaki seq_win_count kadar veriyi cikarmak icin asagidaki kodlari ekledik
    all_pre = []
    for i in range(pre_u.shape[0]):
        p_idx = range(pre_idx[i], pre_idx[i + 1])
        u, idx = np.unique(pre_label[p_idx, 0], return_index=True)
        idx = np.append(idx, np.asarray(p_idx).shape[0])
        part_pre = list(itertools.chain.from_iterable(
            [range(p_idx[idx[j]], p_idx[idx[j + 1]-1]+1 - (seq_win_count-1)) for j in range(idx.shape[0] - 1)]))
        all_pre.append(part_pre)
    all_int = [range(int_idx[i], int_idx[i + 1] - (seq_win_count-1)) for i in range(int_u.shape[0])]

    # preictal kayitlardan under sample ratio oraninda rastgele indexler aliniyor
    for i in range(pre_u.shape[0]):
        np.random.shuffle(all_pre[i])
        all_pre[i]=all_pre[i][0:int(len(all_pre[i]) * under_sample_ratio)]
    train_pre = np.hstack(all_pre)
    np.random.shuffle(train_pre)

    # tum kayitlardaki interictal kayitlar karistiriliyor
    all_int = np.hstack(all_int)
    np.random.shuffle(all_int)
    train_int = all_int[0:len(train_pre)]

    return [train_pre, train_int]

def organize_dataset_balanced_under(images_timewin_pos, images_timewin_neg, seq_win_count=5, s_count=[3, 3],
                                    fold_count=10):
    fold_pairs = []
    images_timewins = []

    for i in range(fold_count):
        # pozitif orneklerden rastgele pencere dizisi secme
        # 3 farkli pencere secilecek
        s_count_pos = s_count[0]
        images_timewin_p = []
        for idx in range(images_timewin_pos.shape[1]):
            w_count = images_timewin_pos.shape[0] / seq_win_count
            w_index = range(w_count)
            np.random.shuffle(w_index)
            for sc in range(s_count_pos):
                images_timewin_p.append(
                    images_timewin_pos[w_index[sc] * seq_win_count:(w_index[sc] + 1) * seq_win_count, idx])
        images_timewin_p = np.swapaxes(np.asarray(images_timewin_p), 0, 1)

        # negatif orneklerden rastgele pencere dizisi secme
        # 3 farkli pencere secilecek
        s_count_neg = s_count[1]
        images_timewin_n = []
        # pozitif ornek kadar negatif ornek secilecek
        w_neg_index = range(images_timewin_neg.shape[1])
        np.random.shuffle(w_neg_index)

        for idx in range(images_timewin_pos.shape[1]):
            w_count = images_timewin_neg.shape[0] / seq_win_count
            w_index = range(w_count)
            np.random.shuffle(w_index)
            for sc in range(s_count_neg):
                images_timewin_n.append(
                    images_timewin_neg[w_index[sc] * seq_win_count:(w_index[sc] + 1) * seq_win_count, w_neg_index[idx]])
        images_timewin_n = np.swapaxes(np.asarray(images_timewin_n), 0, 1)

        # pozitif test indexleri
        index_pos = np.arange(0, images_timewin_p.shape[1])
        np.random.shuffle(index_pos)
        ts_pos = np.asarray(index_pos[:(index_pos.shape[0] / 5)])
        index_pos = np.setdiff1d(index_pos, ts_pos)
        # pozitif valid indexleri
        np.random.shuffle(index_pos)
        vl_pos = index_pos[:(index_pos.shape[0] / 5)]
        # pozitif train indexleri
        tr_pos = np.setdiff1d(index_pos, vl_pos)

        # negatif test indexleri
        index_neg = np.arange(0, images_timewin_n.shape[1])
        np.random.shuffle(index_neg)
        # negatif test sayisi pozitif ile ayni
        ts_neg = np.asarray(index_neg[:(ts_pos.shape[0])])
        index_neg = np.setdiff1d(index_neg, ts_neg)
        # negatif valid indexleri
        np.random.shuffle(index_neg)
        # negatif valid sayisi pozitif ile ayni
        vl_neg = index_neg[:(vl_pos.shape[0])]
        # negatif train indexleri
        tr_neg = np.setdiff1d(index_neg, vl_neg)

        # pozitif ve negatif indeksler kaskad eklenecek
        tr_neg += images_timewin_p.shape[1]
        vl_neg += images_timewin_p.shape[1]
        ts_neg += images_timewin_p.shape[1]

        tr = np.concatenate((tr_pos, tr_neg))
        vl = np.concatenate((vl_pos, vl_neg))
        ts = np.concatenate((ts_pos, ts_neg))

        # train, valid ve test indeksleri shuffle ediliyor
        np.random.shuffle(tr)
        np.random.shuffle(vl)
        np.random.shuffle(ts)

        fold_pairs.append((tr, vl, ts))

        images_timewin = np.concatenate((images_timewin_p, images_timewin_n), axis=1)
        images_timewins.append(images_timewin)

    # labels
    n_labels = np.full([images_timewin_n.shape[1]], 0)
    p_labels = np.full([images_timewin_p.shape[1]], 1)
    labels = np.concatenate((p_labels, n_labels))

    ratio = 1

    return images_timewins, labels, fold_pairs, ratio


def organize_dataset_balanced_over(images_timewin_pos, images_timewin_neg, seq_win_count=5, s_count=[3, 10],
                                   fold_count=10):
    fold_pairs = []
    images_timewins = []

    for i in range(fold_count):
        # pozitif orneklerden rastgele pencere dizisi secme
        # 3 farkli pencere secilecek
        s_count_pos = s_count[0]
        images_timewin_p = []
        for idx in range(images_timewin_pos.shape[1]):
            w_count = images_timewin_pos.shape[0] / seq_win_count
            w_index = range(w_count)
            np.random.shuffle(w_index)
            for sc in range(s_count_pos):
                images_timewin_p.append(
                    images_timewin_pos[w_index[sc] * seq_win_count:(w_index[sc] + 1) * seq_win_count, idx])
        images_timewin_p = np.swapaxes(np.asarray(images_timewin_p), 0, 1)

        # negatif orneklerden rastgele pencere dizisi secme
        # 10 farkli pencere secilecek
        s_count_neg = s_count[1]
        images_timewin_n = []
        for idx in range(images_timewin_neg.shape[1]):
            w_count = images_timewin_neg.shape[0] / seq_win_count
            w_index = range(w_count)
            np.random.shuffle(w_index)
            for sc in range(s_count_neg):
                images_timewin_n.append(
                    images_timewin_neg[w_index[sc] * seq_win_count:(w_index[sc] + 1) * seq_win_count, idx])
        images_timewin_n = np.swapaxes(np.asarray(images_timewin_n), 0, 1)

        # pozitif test indexleri
        index_pos = np.arange(0, images_timewin_p.shape[1])
        np.random.shuffle(index_pos)
        ts_pos = np.asarray(index_pos[:(index_pos.shape[0] / 5)])
        index_pos = np.setdiff1d(index_pos, ts_pos)
        # pozitif valid indexleri
        np.random.shuffle(index_pos)
        vl_pos = index_pos[:(index_pos.shape[0] / 5)]
        # pozitif train indexleri
        tr_pos = np.setdiff1d(index_pos, vl_pos)

        # pozitif ornek indexleri oversample ediliyor
        d_count = images_timewin_n.shape[1] / images_timewin_p.shape[1]
        ts_pos_over = []
        vl_pos_over = []
        tr_pos_over = []
        for i in range(d_count):
            ts_pos_over = np.concatenate((ts_pos_over, ts_pos))
            vl_pos_over = np.concatenate((vl_pos_over, vl_pos))
            tr_pos_over = np.concatenate((tr_pos_over, tr_pos))
        ts_pos = ts_pos_over.astype(np.int32)
        vl_pos = vl_pos_over.astype(np.int32)
        tr_pos = tr_pos_over.astype(np.int32)

        # negatif test indexleri
        index_neg = np.arange(0, images_timewin_n.shape[1])
        np.random.shuffle(index_neg)
        # negatif test sayisi pozitif ile ayni
        ts_neg = np.asarray(index_neg[:(ts_pos.shape[0])])
        index_neg = np.setdiff1d(index_neg, ts_neg)
        # negatif valid indexleri
        np.random.shuffle(index_neg)
        # negatif valid sayisi pozitif ile ayni
        vl_neg = index_neg[:(vl_pos.shape[0])]
        # negatif train indexleri
        tr_neg = np.setdiff1d(index_neg, vl_neg)

        # pozitif ve negatif indeksler kaskad eklenecek
        tr_neg += images_timewin_p.shape[1]
        vl_neg += images_timewin_p.shape[1]
        ts_neg += images_timewin_p.shape[1]

        tr = np.concatenate((tr_pos, tr_neg))
        vl = np.concatenate((vl_pos, vl_neg))
        ts = np.concatenate((ts_pos, ts_neg))

        # train, valid ve test indeksleri shuffle ediliyor
        np.random.shuffle(tr)
        np.random.shuffle(vl)
        np.random.shuffle(ts)

        fold_pairs.append((tr, vl, ts))

        images_timewin = np.concatenate((images_timewin_p, images_timewin_n), axis=1)
        images_timewins.append(images_timewin)

    # labels
    n_labels = np.full([images_timewin_n.shape[1]], 0)
    p_labels = np.full([images_timewin_p.shape[1]], 1)
    labels = np.concatenate((p_labels, n_labels))

    ratio = 1

    return images_timewins, labels, fold_pairs, ratio
