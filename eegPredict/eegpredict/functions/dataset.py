# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import scipy.io

from eegpredict.functions.utils import azim_proj, EmptyClass, standardDataAdaptiveSegment, timeDiff, standardDataAll
from eegpredict.functions.gen_images import gen_image_dataset, organize_dataset_imbalanced_conf, \
    organize_dataset_pretrain
from eegpredict.functions import globals

# globals.dataset_link='/Volumes/MacHDD/Dataset/physiobank/chbmit/'
globals.dataset_link = 'D:/Dataset/physiobank/chbmit/'
globals.seizureList_file = '../../generateEEGFeats/refData/seizureList.mat'
globals.records_file = '../../generateEEGFeats/refData/records.mat'

def load_locations(withProjection=True):
    print('Electrode locations is loading...')
    if withProjection:
        locs = scipy.io.loadmat('../../generateEEGFeats/refData/10_20_eeg_locs.mat')
        locs_3d = locs['locs3d']
        locs_2d = []

        # 2D Projection
        for e in locs_3d:
            locs_2d.append(azim_proj(e))

        # Shift to zero
        locs_2d = np.asarray(locs_2d)
        min_0 = locs_2d[:, 0].min()
        min_1 = locs_2d[:, 1].min()

        locs_2d = locs_2d - (min_0, min_1)

    else:
        locs = scipy.io.loadmat('../../generateEEGFeats/refData/locs2d_20c.mat')
        locs_2d = locs['locs2d']
    return locs_2d


def init_feature_types(psd=True, moment=True, hjorth=True, alg_comp=True):

    feature_parameters = EmptyClass()

    # Features
    feature_psd = range(0, 8)
    feature_moment = range(8, 12)
    feature_hjorth = range(12, 14)
    feature_alg_comp = range(14, 15)

    eeg_channel_count = 20

    feature_eval = []
    if psd:
        feature_eval = np.hstack([feature_eval, feature_psd])

    if moment:
        feature_eval = np.hstack([feature_eval, feature_moment])

    if hjorth:
        feature_eval = np.hstack([feature_eval, feature_hjorth])

    if alg_comp:
        feature_eval = np.hstack([feature_eval, feature_alg_comp])

    feature_parameters.feature_eval = feature_eval.astype(int)

    # feature size =  14 feature x 20 channel
    feature_parameters.feature_count = np.asarray(feature_eval).shape[0]
    feature_parameters.feature_size_all = (len(feature_psd)+len(feature_moment)+len(feature_hjorth)+len(feature_alg_comp)) * eeg_channel_count

    return feature_parameters


def prepare_dataset(feat_params, model_params, pat_id, gen_images=False, under_sample=True, under_sample_ratio=1, normalization=False, time_diff=False):
    dataset = EmptyClass()
    dataset.pat_id = pat_id
    dataset.under_sample = under_sample
    dataset.under_sample_ratio = under_sample_ratio
    dataset.num_fold = 5

    # Feature dataset
    feature_info = scipy.io.loadmat('../../generateEEGFeats/dataset/pat_' + str(pat_id) + '_info.mat')['pat_info']

    if gen_images:
        # generating image data
        gen_image_dataset(model_params.locs_2d, pat_id, feature_info, feature_size=feat_params.feature_size_all,
                          pixelCount=model_params.pixel_count, gen_image=model_params.gen_image_type)

    # images are loading
    images_timewin_pre = scipy.io.loadmat('../data/pat_' + str(pat_id) + '_pre_' + str(model_params.pixel_count[1]) + 'px.mat')['images_timewin_pre']
    pre_label = scipy.io.loadmat('../data/pat_' + str(pat_id) + '_pre_label.mat')['pat_pre_lbl']
    images_timewin_int = scipy.io.loadmat('../data/pat_' + str(pat_id) + '_int_' + str(model_params.pixel_count[1]) + 'px.mat')['images_timewin_int']
    int_label = scipy.io.loadmat('../data/pat_' + str(pat_id) + '_int_label.mat')['pat_int_lbl']


    fold_pairs, ratio = organize_dataset_imbalanced_conf(pre_label, int_label, seq_win_count=model_params.seq_win_count,
                                                         fold_count=dataset.num_fold, feature_info=feature_info[0][0],
                                                         preictal_with_int_pair=True, train_valid_method='split_fold')

    # standardization
    if normalization:
        images_timewin_pre = standardDataAdaptiveSegment(images_timewin_pre, pre_label[:, 1], with_mean=False, norm_window_len=30)
        images_timewin_int = standardDataAdaptiveSegment(images_timewin_int, int_label[:, 0], with_mean=False, norm_window_len=30)


    # time differential feature
    if time_diff:
        images_timewin_pre = timeDiff(images_timewin_pre, pre_label[:, 1])
        images_timewin_int = timeDiff(images_timewin_int, int_label[:, 0])



    images_timewin = []
    images_timewin.append((images_timewin_pre[:, feat_params.feature_eval], images_timewin_int[:, feat_params.feature_eval]))

    dataset.images = images_timewin
    dataset.fold_pairs = fold_pairs
    dataset.class_weights = [1, ratio]
    dataset.feature_info = feature_info[0][0]
    dataset.feat_params = feat_params

    return dataset

def prepare_pre_dataset(feat_params, model_params, pat_ids, under_sample=True, under_sample_ratio=0.1, normalization=False, time_diff=False):
    dataset = EmptyClass()
    dataset.pat_ids = pat_ids
    dataset.under_sample = under_sample
    dataset.under_sample_ratio = under_sample_ratio
    dataset.num_fold = 5
    dataset.class_weights = [1, 1]

    images_timewin_pre_all = []
    images_timewin_int_all = []
    for i in pat_ids:
        # Feature dataset
        feature_info = scipy.io.loadmat('../../generateEEGFeats/dataset/pat_' + str(i) + '_info.mat')['pat_info']

        # images are loading
        images_timewin_pre = scipy.io.loadmat('../data/pat_' + str(i) + '_pre_' + str(model_params.pixel_count[1]) + 'px.mat')['images_timewin_pre']
        pre_label = scipy.io.loadmat('../data/pat_' + str(i) + '_pre_label.mat')['pat_pre_lbl']
        images_timewin_int = scipy.io.loadmat('../data/pat_' + str(i) + '_int_' + str(model_params.pixel_count[1]) + 'px.mat')['images_timewin_int']
        int_label = scipy.io.loadmat('../data/pat_' + str(i) + '_int_label.mat')['pat_int_lbl']


        train_pairs = organize_dataset_pretrain(pre_label, int_label, seq_win_count=model_params.seq_win_count,
                                                      under_sample_ratio=under_sample_ratio)

        # standardization
        if normalization:
            images_timewin_pre, images_timewin_int = standardDataAll(images_timewin_pre, images_timewin_int,
                                                                     with_mean=True)

        # time differential feature
        if time_diff:
            images_timewin_pre = timeDiff(images_timewin_pre, pre_label[:, 1])
            images_timewin_int = timeDiff(images_timewin_int, int_label[:, 0])

        train_pre = []
        train_int = []
        for i in range(len(train_pairs[0])):
            train_pre.append(np.expand_dims(images_timewin_pre[train_pairs[0][i]:train_pairs[0][i] + model_params.seq_win_count,
            feat_params.feature_eval], axis=0))
            train_int.append(np.expand_dims(images_timewin_int[train_pairs[1][i]:train_pairs[1][i] + model_params.seq_win_count,
            feat_params.feature_eval], axis=0))
        train_pre = np.squeeze(np.vstack(train_pre))
        train_int = np.squeeze(np.vstack(train_int))

        if model_params.seq_win_count > 1:
            train_pre = np.swapaxes(train_pre, 1, 2)
            train_int = np.swapaxes(train_int, 1, 2)

        images_timewin_pre_all.append(train_pre)
        images_timewin_int_all.append(train_int)

    images_timewin_pre_all = np.vstack(images_timewin_pre_all)
    images_timewin_int_all = np.vstack(images_timewin_int_all)
    images_timewin = [images_timewin_pre_all, images_timewin_int_all]
    dataset.images = images_timewin

    return dataset