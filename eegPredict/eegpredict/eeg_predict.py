# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import scipy.io

from eegpredict.functions.eeg_predict_utils import init_model_parameters, init_train_parameters
from eegpredict.functions.plot_data import plot_eeg_prediction
from eegpredict.functions.train import create_model, train_model, pre_train_model
from eegpredict.functions.utils import analysis_cnn_model

from eegpredict.functions.dataset import init_feature_types, prepare_dataset, prepare_pre_dataset

# deterministic random
np.random.seed(1234)


def do_job(job, job_id, pat_id, model_params, feat_params, train_params):

    if job == "pretrain":

        # All Patients
        all_patiens = range(1, 25)

        # Dataset hazirlaniyor
        pat_ids = np.setdiff1d(all_patiens, pat_id)
        pre_dataset = prepare_pre_dataset(feat_params, model_params, pat_ids, under_sample=True, under_sample_ratio=0.1,
                                          normalization=True)

        # Model olusturuluyor
        model = create_model(model_params, feat_params, train_params, pre_dataset)

        # On Egitim
        pre_train_model(model, pre_dataset)

    elif job == "train":
        # Dataset hazirlaniyor
        dataset = prepare_dataset(feat_params, model_params, pat_id, gen_images=False, under_sample=True,
                                  under_sample_ratio=1)

        # Model olusturuluyor
        model = create_model(model_params, feat_params, train_params, dataset)

        # Model analiz ediliyor
        analysis_cnn_model(model)

        # Model egitiliyor
        train_result, test_result, predict_result = train_model(model, dataset)

        # Sonuclar kaydediliyor
        save_idx = job_id
        scipy.io.savemat('../test/' + str(save_idx) + '_pat_' + str(pat_id) + '_train_result_' + str(model_params.model_name) + '.mat',
                         {'train_result': train_result})
        scipy.io.savemat('../test/' + str(save_idx) + '_pat_' + str(pat_id) + '_test_result_' + str(model_params.model_name) + '.mat',
                         {'test_result': test_result})
        scipy.io.savemat('../test/' + str(save_idx) + '_pat_' + str(pat_id) + '_predict_result_' + str(model_params.model_name) + '.mat',
                     {'predict_result': predict_result})

    elif job == "plot":

        # Dataset hazirlaniyor
        dataset = prepare_dataset(feat_params, model_params, pat_id, gen_images=False, under_sample=True,
                                  under_sample_ratio=1)

        load_idx = job_id
        predict_result = scipy.io.loadmat('../test/' + str(load_idx) + '_pat_' + str(pat_id) + '_predict_result_' +
                                          str(model_params.model_name) + '.mat')['predict_result']

        plot_eeg_prediction(dataset, predict_result, window_length=4, overlap=0.5)

    elif job == "gen_images":

        # All Patients
        all_patiens = range(1, 25)

        for i in all_patiens:
            try:
                prepare_dataset(feat_params, model_params, i, gen_images=True, under_sample=True,
                                under_sample_ratio=1)
            except:
                continue


if __name__ == '__main__':



    # Jobs: pretrain, train, plot, gen_images



    # model_name =  model_cnn, model_cnn3d, model_cnn_max, model_cnn_conv1d, model_cnn_lstm, model_cnn_mix,
    #               model_cnn_lstm_hybrid,
    #               model_custom_mlp, model_custom_mlp_multi,
    #               model_svm

    do_job(job="train",
            job_id=1001,
            pat_id=2,
            model_params=init_model_parameters(model_name='model_cnn3d'),
            train_params=init_train_parameters(num_epoch=50, batch_size=100, l1=0.0, l2=0.035, learn_rate=0.001,
                                               online_weights=False, save_model_params=False, plot_prediction=True),
            feat_params=init_feature_types(psd=True, moment=True, hjorth=True, alg_comp=True))



