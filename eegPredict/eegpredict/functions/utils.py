__author__ = 'Ahmet Remzi Ozcan, Pouya Bashivan'

import time
import math as m
import numpy as np
import scipy.io
import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, Normalizer, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt


class EmptyClass(object):
    pass


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def find_split_idx(data):
    idx=[0]
    for i in range(1,len(data)):
        if data[i] != data[i-1]+1:
            idx.append(i)
    return idx


def print_dashed_line():
    print("----------------------------------------------------------------------------------")


def calc_mean(pre, int, pre_len, int_len, type='basic'):
    # Basit Ortalama
    if type == 'basic':
        mean = (pre + int) / 2
    elif type == 'weighted':
        mean = (pre * pre_len + int * int_len) / (pre_len + int_len)
    elif type == 'int_only':
        mean = float(int)
    return mean

def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x ** 2 + y ** 2
    r = m.sqrt(x2_y2 + z ** 2)  # r
    elev = m.atan2(z, m.sqrt(x2_y2))  # Elevation
    az = m.atan2(y, x)  # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * m.cos(theta), rho * m.sin(theta)


def augment_EEG(data, stdMult, pca=False, n_components=2):
    """
    Augment data by adding normal noise to each feature.

    :param data: EEG feature data as a matrix (n_samples x n_features)
    :param stdMult: Multiplier for std of added noise
    :param pca: if True will perform PCA on data and add noise proportional to PCA components.
    :param n_components: Number of components to consider when using PCA.
    :return: Augmented data as a matrix (n_samples x n_features)
    """
    augData = np.zeros(data.shape)
    if pca:
        pca = PCA(n_components=n_components)
        pca.fit(data)
        components = pca.components_
        variances = pca.explained_variance_ratio_
        coeffs = np.random.normal(scale=stdMult, size=pca.n_components) * variances
        for s, sample in enumerate(data):
            augData[s, :] = sample + (components * coeffs.reshape((n_components, -1))).sum(axis=0)
    else:
        # Add Gaussian noise with std determined by weighted std of each feature
        for f, feat in enumerate(data.transpose()):
            augData[:, f] = feat + np.random.normal(scale=stdMult * np.std(feat), size=feat.size)
    return augData


def augment_EEG_image(image, std_mult, pca=False, n_components=2):
    """
    Augment data by adding normal noise to each feature.

    :param image: EEG feature data as a a colored image [n_samples, n_colors, W, H]
    :param std_mult: Multiplier for std of added noise
    :param pca: if True will perform PCA on data and add noise proportional to PCA components.
    :param n_components: Number of components to consider when using PCA.
    :return: Augmented data as a matrix (n_samples x n_features)
    """
    augData = np.zeros((data.shape[0], data.shape[1], data.shape[2] * data.shape[3]))
    for c in xrange(image.shape[1]):
        reshData = np.reshape(data['featMat'][:, c, :, :], (data['featMat'].shape[0], -1))
        if pca:
            augData[:, c, :] = augment_EEG(reshData, std_mult, pca=True, n_components=n_components)
        else:
            augData[:, c, :] = augment_EEG(reshData, std_mult, pca=False)
    return np.reshape(augData, data['featMat'].shape)


def load_data(data_file):
    """
    Loads the data from MAT file. MAT file should contain two
    variables. 'featMat' which contains the feature matrix in the
    shape of [samples, features] and 'labels' which contains the output
    labels as a vector. Label numbers are assumed to start from 1.

    Parameters
    ----------
    data_file: str

    Returns
    -------
    data: array_like
    """
    print("Loading data from %s" % (data_file))

    dataMat = scipy.io.loadmat(data_file, mat_dtype=True)

    print("Data loading complete. Shape is %r" % (dataMat['featMat'].shape,))
    return dataMat['features'][:, :-1], dataMat['features'][:, -1] - 1  # Sequential indices


def reformatInput(data, indices, f, seq_win_count, under_sample=False, under_sample_ratio=1, model_name="model_cnn",
                  train_weights=True):
    """
    Receives the the indices for train and test datasets.
    Outputs the train, validation, and test data and label datasets.
    Parameters
    ----------
    data:           1 x 2 x 4D array    ---     1 x (preictal,interictal) x (sample x feature x row x col)
    indices:        (preictal,interictal) x (fold) x (sample)
    f:              fold number
    seq_win_count:  sequential window count
    underSample:    undersample flag in the train data to balance unbalanced positive and negative samples

    Returns
    -------
    trainData, trainLabels
    validData, validLabels
    testData, testLabels

    """

    pre_label = 1
    int_label = 0

    # datasette nan ifadeler varsa numerik degerlerle degistirilecek
    data_pre = np.nan_to_num(data[0][0])
    data_int = np.nan_to_num(data[0][1])

    # standard data
    # once train data
    # data_pre_train_idx = np.asarray(list(itertools.chain.from_iterable(indices[0])))
    # data_pre_train_idx.sort()
    # data_pre[data_pre_train_idx] = standardDataAdaptive(data_pre[data_pre_train_idx], with_mean=False,
    #                                                     norm_window_len=30)
    # data_int_train_idx = np.asarray(list(itertools.chain.from_iterable(indices[1])))
    # data_int_train_idx.sort()
    # data_int[data_int_train_idx] = standardDataAdaptive(data_int[data_int_train_idx], with_mean=False,
    #                                                     norm_window_len=30)
    # data_pre_test_idx = indices[2]
    # data_pre_test_idx.sort()
    # data_pre[data_pre_test_idx] = standardDataAdaptive(data_pre[data_pre_test_idx], with_mean=False,
    #                                                     norm_window_len=100)
    # data_int_test_idx = indices[3]
    # data_int_test_idx.sort()
    # data_int[data_int_test_idx] = standardDataAdaptive(data_int[data_int_test_idx], with_mean=False,
    #                                                     norm_window_len=100)

    # validIndices_pre = indices[0][f]
    # trainIndices_pre = np.setdiff1d(list(itertools.chain.from_iterable(indices[0])), validIndices_pre)
    trainIndices_pre = indices[0][f][0]
    validIndices_pre = indices[0][f][1]
    # np.random.shuffle(trainIndices_pre)
    # np.random.shuffle(validIndices_pre)

    # validIndices_int = indices[1][f]
    # trainIndices_int = np.setdiff1d(list(itertools.chain.from_iterable(indices[1])), validIndices_int)
    trainIndices_int = indices[1][f][0]
    validIndices_int = indices[1][f][1]
    # np.random.shuffle(trainIndices_int)
    # np.random.shuffle(validIndices_int)

    if train_weights:
        pre_range = [range(indices[5][i+1]-indices[5][i]) for i in range(len(indices[5])-1)]
        pre_weights = np.hstack([np.asarray(i, dtype=float)/(len(i)/2) for i in pre_range])
        int_weights = np.ones(len(data_int))

    if under_sample:
        trainIndices_int = trainIndices_int[range(trainIndices_pre.shape[0]*under_sample_ratio)]
        np.random.shuffle(validIndices_int)
        np.random.shuffle(validIndices_pre)
        validIndices_int = validIndices_int[range(validIndices_pre.shape[0]*under_sample_ratio)]



    trainLabels = np.concatenate(
        (np.full((trainIndices_pre.shape[0]), pre_label).astype(np.int32),
         np.full((trainIndices_int.shape[0]), int_label).astype(np.int32)), axis=0)

    validLabels = [np.full((validIndices_pre.shape[0]), pre_label).astype(np.int32),
                   np.full((validIndices_int.shape[0]), int_label).astype(np.int32)]

    testIndices_pre = indices[2]
    testIndices_int = indices[3]
    testLabels = [np.full((testIndices_pre.shape[0]), pre_label).astype(np.int32),
                  np.full((testIndices_int.shape[0]), int_label).astype(np.int32)]

    # tek bir pencere icin islem yapiliyorsa
    if seq_win_count == 1:
        trainData = np.concatenate((data_pre[trainIndices_pre], data_int[trainIndices_int]), axis=0)
        trainWeights = np.concatenate((pre_weights[trainIndices_pre], int_weights[trainIndices_int]), axis=0)
        trainIndices = range(trainData.shape[0])
        np.random.shuffle(trainIndices)

        validData = [data_pre[validIndices_pre], data_int[validIndices_int]]
        testData = [data_pre[testIndices_pre], data_int[testIndices_int]]

        return [(trainData[trainIndices], trainLabels[trainIndices].astype(np.int32), trainWeights[trainIndices]),
                (validData, validLabels),
                (testData, testLabels)]

    # birden fazla pencere varsa
    elif seq_win_count > 1:
        trainData = np.asarray(
            [np.concatenate((data_pre[trainIndices_pre + i], data_int[trainIndices_int + i]), axis=0) for i in
             range(seq_win_count)])
        trainWeights = np.concatenate((pre_weights[trainIndices_pre], int_weights[trainIndices_int]), axis=0)
        trainIndices = range(trainData.shape[1])
        np.random.shuffle(trainIndices)

        validData = [np.asarray([data_pre[validIndices_pre + i] for i in range(seq_win_count)]),
                     np.asarray([data_int[validIndices_int + i] for i in range(seq_win_count)])]

        testData = [np.asarray([data_pre[testIndices_pre + i] for i in range(seq_win_count)]),
                    np.asarray([data_int[testIndices_int + i] for i in range(seq_win_count)])]

        if model_name == 'model_cnn3d' or model_name =='model_cnn3d_basic' or model_name =='model_cnn_temp':
            trainData, validData[0], validData[1], testData[0], testData[1] = \
                swapDataAxes((trainData, validData[0], validData[1], testData[0], testData[1]), 0, 1)
            trainData, validData[0], validData[1], testData[0], testData[1] = \
                swapDataAxes((trainData, validData[0], validData[1], testData[0], testData[1]), 1, 2)
            return [(trainData[trainIndices], trainLabels[trainIndices].astype(np.int32), trainWeights[trainIndices]),
                    (validData, validLabels),
                    (testData, testLabels)]
        else:
            return [(trainData[:, trainIndices], trainLabels[trainIndices].astype(np.int32), trainWeights[trainIndices]),
                    (validData, validLabels),
                    (testData, testLabels)]


def swapDataAxes(data, ind1, ind2):
    swappedData = []
    for i in data:
        swappedData.append(np.swapaxes(np.asarray(i), ind1, ind2))
    return swappedData


def transformData(data, sc):
    transformedData = []
    for d in data:
        transformedData.append(sc.transform(d.reshape(-1, 1)).reshape(d.shape))
    return transformedData


def normalizeData(data):
    if data.ndim == 4:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = normalize(data[i, j])
    elif data.ndim == 5:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    data[i, j, k] = normalize(data[i, j, k])
    return data


def timeDiff(data, label):

    u, idx = np.unique(label, return_index=True)
    idx = np.append(idx, label.shape[0])

    # oznitelik boyunca farki bulabilmek icin boyut sirasini degistiriyoruz
    data = np.swapaxes(data, 0, 1)
    new_data = []
    for i in range(data.shape[0]):
        diff_k = []
        for k in range(len(idx)-1):
            diff = np.array([np.subtract(data[i][n+1], data[i][n]) for n in range(idx[k], idx[k+1]-1)])
            diff = np.concatenate((diff, [diff[-1]]))
            diff_k = diff if diff_k == [] else np.concatenate((diff_k, diff))
        data[i] = diff_k
        #new_data.append(diff_k)
    # boyut sirasini eski haline getiriyoruz
    # data = np.concatenate((data, new_data))
    data = np.swapaxes(data, 0, 1)
    return data


def standardDataAdaptiveSegment(data, label, with_mean=False, norm_window_len=100):


    u, idx = np.unique(label, return_index=True)
    idx = np.append(idx, label.shape[0])

    # oznitelik boyunca standartizasyon yapabilek icin boyut sirasini degistiriyoruz
    data = np.swapaxes(data, 0, 1)
    for i in range(data.shape[0]):
        for k in range(len(idx)-1):
            sc = StandardScaler(with_mean=with_mean)
            sc_pre = StandardScaler(with_mean=with_mean)
            data_pair_idx = np.asarray(list(chunks(range(idx[k], idx[k+1]), norm_window_len)))
            for j in data_pair_idx:
                # standartizasyon parametreleri hesaplaniyor
                sc.fit(data[i][j].reshape(-1, 1))

                sc_pre.partial_fit(data[i][j].reshape(-1, 1))

                # veriler standartize ediliyor
                data[i][j] = sc_pre.transform(data[i][j].reshape(-1, 1)).reshape(data[i][j].shape)

                sc_pre.mean_ = sc.mean_
                sc_pre.scale_ = sc.scale_
    # boyut sirasini eski haline getiriyoruz
    data = np.swapaxes(data, 0, 1)
    return data


def standardDataAdaptive(data, with_mean=False, norm_window_len=100):

    # oznitelik boyunca standartizasyon yapabilek icin boyut sirasini degistiriyoruz
    data = np.swapaxes(data, 0, 1)
    for i in range(data.shape[0]):
        sc = StandardScaler(with_mean=with_mean)
        sc_pre = StandardScaler(with_mean=with_mean)
        data_pair_idx = np.asarray(list(chunks(range(data.shape[1]), norm_window_len)))
        for j in data_pair_idx:
            # standartizasyon parametreleri hesaplaniyor
            sc.fit(data[i][j].reshape(-1, 1))

            sc_pre.partial_fit(data[i][j].reshape(-1, 1))

            # veriler standartize ediliyor
            data[i][j] = sc_pre.transform(data[i][j].reshape(-1, 1)).reshape(data[i][j].shape)

            sc_pre.mean_ = sc.mean_
            sc_pre.scale_ = sc.scale_

    # boyut sirasini eski haline getiriyoruz
    data = np.swapaxes(data, 0, 1)
    return data

def standardDataAll(pre_data,int_data, with_mean=False):

    # oznitelik boyunca standartizasyon yapabilek icin boyut sirasini degistiriyoruz
    all_data = np.concatenate((pre_data, int_data), axis=0)
    pre_data = np.swapaxes(pre_data, 0, 1)
    int_data = np.swapaxes(int_data, 0, 1)
    all_data = np.swapaxes(all_data, 0, 1)
    for i in range(all_data.shape[0]):
        sc = StandardScaler(with_mean=with_mean)

        # standartizasyon parametreleri hesaplaniyor
        sc.fit(all_data[i].reshape(-1, 1))

        # veriler standartize ediliyor
        pre_data[i] = sc.transform(pre_data[i].reshape(-1, 1)).reshape(pre_data[i].shape)
        int_data[i] = sc.transform(int_data[i].reshape(-1, 1)).reshape(int_data[i].shape)

    # boyut sirasini eski haline getiriyoruz
    pre_data = np.swapaxes(pre_data, 0, 1)
    int_data = np.swapaxes(int_data, 0, 1)
    return pre_data, int_data


def standardData(trainData, validData, testData, with_mean=False, in_window_norm=False, in_feat_norm=True,
                 model_name='model_cnn', normalize=False):
    if in_feat_norm:
        if normalize:
            sc = Normalizer()
        else:
            sc = StandardScaler(with_mean=with_mean)

        if trainData.ndim == 4 or model_name == 'model_cnn3d' or model_name =='model_cnn3d_basic' or model_name =='model_cnn_temp':
            #trainAll = np.concatenate((trainData, validData[0], validData[1]), axis=0)
            trainAll = trainData
            # oznitelik boyunca standartizasyon yapabilek icin boyut sirasini degistiriyoruz
            trainAll, trainData, validData[0], validData[1], testData[0], testData[1] = \
                swapDataAxes((trainAll, trainData, validData[0], validData[1], testData[0], testData[1]), 0, 1)
            for i in range(trainAll.shape[0]):
                # train verilerine gore standartizasyon parametreleri hesaplandi
                sc.fit(trainAll[i].reshape(-1, 1))
                # train verileri standartize ediliyor
                trainData[i], validData[0][i], validData[1][i] = \
                    transformData((trainData[i], validData[0][i], validData[1][i]), sc)
                # test verileri standartize ediliyor
                testData[0][i], testData[1][i] = transformData((testData[0][i], testData[1][i]), sc)

            # boyut sirasini eski haline getiriyoruz
            trainData, validData[0], validData[1], testData[0], testData[1] = \
                swapDataAxes((trainData, validData[0], validData[1], testData[0], testData[1]), 0, 1)

        elif trainData.ndim == 5:
            #trainAll = np.concatenate((trainData, validData[0], validData[1]), axis=1)
            trainAll = trainData
            # oznitelik boyunca standartizasyon yapabilek icin boyut sirasini degistiriyoruz
            trainAll, trainData, validData[0], validData[1], testData[0], testData[1] = \
                swapDataAxes((trainAll, trainData, validData[0], validData[1], testData[0], testData[1]), 0, 2)
            for i in range(trainAll.shape[0]):
                # train verilerine gore standartizasyon parametreleri hesaplandi
                sc.fit(trainAll[i].reshape(-1, 1))
                # train verileri standartize ediliyor
                trainData[i], validData[0][i], validData[1][i] = \
                    transformData((trainData[i], validData[0][i], validData[1][i]), sc)
                # test verileri standartize ediliyor
                testData[0][i], testData[1][i] = transformData((testData[0][i], testData[1][i]), sc)
            # boyut sirasini eski haline getiriyoruz
            trainData, validData[0], validData[1], testData[0], testData[1] = \
                swapDataAxes((trainData, validData[0], validData[1], testData[0], testData[1]), 0, 2)

    if in_window_norm:
        trainData = normalizeData(trainData)
        validData[0] = normalizeData(validData[0])
        validData[1] = normalizeData(validData[1])
        testData[0] = normalizeData(testData[0])
        testData[1] = normalizeData(testData[1])

    return trainData, validData, testData


def moving_average(a, n=3):
    for i in range(n - 1):
        a = np.insert(a, 0, a[0])
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_svm_coefficients(coef, label, title="Features"):
    coef = np.asarray(coef)
    plt.figure(figsize=(15, 5))
    plt.boxplot(coef, showmeans=True, whis=99)
    plt.xticks(np.arange(1, coef.shape[1]+1), np.asarray(label))
    plt.xlabel(title)
    plt.title("Coefficients of " + title + " in Linear SVM")
    plt.grid(True, axis='y',linestyle='--')
    plt.show()


def score_svm_coefficients(classifier, asType="all", channel_count=20, top_features=40, isSorted=False, show=True):
    coef_in = classifier.coef_.ravel()
    feature_count = len(coef_in) / channel_count
    if asType == "feature":
        coef = []
        for i in range(0, len(coef_in), channel_count):
            coef.append(sum(np.abs(coef_in[i:i + channel_count])))
        coef = np.asarray(coef)
        label = range(1, feature_count + 1)
        title = "Features"
        if isSorted:
            top_coefficients = np.argsort(coef)
        else:
            top_coefficients = range(len(coef))
        top_features = len(top_coefficients)
    elif asType == "channel":
        coef = np.zeros(channel_count)
        for i in range(0, len(coef_in), 20):
            coef = coef + np.abs(coef_in[i:i + channel_count])
        coef = np.asarray(coef)
        label = range(1, channel_count + 1)
        title = "Channels"
        if isSorted:
            top_coefficients = np.argsort(coef)
        else:
            top_coefficients = range(len(coef))
        top_features = len(top_coefficients)
    else:
        coef = coef_in
        top_coefficients = np.argsort(np.abs(coef))[-top_features:]
        label = []
        title = "Channels and Features"
        for i in range(feature_count):
            for j in range(channel_count):
                label.append("f:" + str(i + 1) + " c:" + str(j + 1))

    # create plot
    if show:
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(top_features), np.abs(coef[top_coefficients]), color=colors)

        plt.xticks(np.arange(0, top_features), np.asarray(label)[top_coefficients], rotation=90, ha='right')
        plt.title("Top " + title + " in Linear SVM")
        plt.show()
    return top_coefficients, coef[top_coefficients], np.asarray(label)[top_coefficients]


def analysis_cnn_model(model):

    from nolearn.lasagne import PrintLayerInfo

    layer_info = PrintLayerInfo()

    layer_info(model.model)


def cross_entropy_loss(predictions, targets):
    if targets:
        return -np.log(predictions)
    else:
        return -np.log(1.0 - predictions)

def sigmoid_entropy_loss(predictions, targets, treshold=0.8, beta=1, alfa=20):
    if targets:
        return beta / (1 + np.exp(-alfa * (treshold - predictions)))
    else:
        return beta / (1 + np.exp(-alfa * (predictions - treshold)))

def calc_avg_val_loss(pred_pre,pred_int,valid_avg_size=30, ratio=1):
    pred_pre = moving_average(pred_pre, valid_avg_size)
    pred_int = np.hstack([np.mean(i) for i in chunks(pred_int,valid_avg_size)])
    pred_pre = np.sort(pred_pre)[::-1]
    pred_pre = pred_pre[0:int(len(pred_pre)*ratio)]
    pre_loss = np.mean(cross_entropy_loss(pred_pre, 1))
    int_loss = np.mean(cross_entropy_loss(pred_int, 0))
    loss = np.mean([pre_loss,int_loss])
    return loss


