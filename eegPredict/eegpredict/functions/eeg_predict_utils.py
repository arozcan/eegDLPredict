# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import scipy.io

from eegpredict.functions.dataset import load_locations
from eegpredict.functions.utils import EmptyClass


def init_train_parameters(num_epoch=50, batch_size=100, l1=0.0, l2=0.001, learn_rate=0.001, online_weights=False,
                          save_model_params=False, plot_prediction=False):
    train_parameters = EmptyClass()

    # number of train epoch
    train_parameters.num_epoch = num_epoch

    # train mini batch size
    train_parameters.batch_size = batch_size

    # save model parameters
    train_parameters.save_model_params = save_model_params

    # init weights with pretrained params
    train_parameters.pre_trained_init = False

    # l1 regularization parameter
    train_parameters.l1 = l1

    # l2 regularization parameter
    train_parameters.l2 = l2

    # learn rate
    train_parameters.learn_rate = learn_rate

    # variable train weights
    train_parameters.online_weights = online_weights

    # plot prediction value on training
    train_parameters.plot_prediction = plot_prediction

    return train_parameters


def init_model_parameters(model_name):
    model_parameters = EmptyClass()
    model_parameters.model_name = model_name
    if model_name == 'model_svm':

        # goruntu boyutu
        model_parameters.pixel_count = [4, 5]

        # ardisil pencere sayisi
        model_parameters.seq_win_count = 1

        # Elektrot yerlesimleri yukleniyor
        model_parameters.locs_2d = load_locations(withProjection=False)

        # Imge olusturma yontemi
        model_parameters.gen_image_type = 'basic'

        # svm kernel type
        model_parameters.kernel = 'rbf'

    elif model_name == 'model_custom_mlp':

        # goruntu boyutu
        model_parameters.pixel_count = [4, 5]

        # ardisil pencere sayisi
        model_parameters.seq_win_count = 1

        # Elektrot yerlesimleri yukleniyor
        model_parameters.locs_2d = load_locations(withProjection=False)

        # Imge olusturma yontemi
        model_parameters.gen_image_type = 'basic'

        # Tam Bagli Katman Hucre Sayisi
        model_parameters.dense_num_unit = [512,512,512]

        # Tam Bagli Katman Hucre Sayisi
        model_parameters.dense_num_unit = [512, 512, 512]

        # Giriste dropout
        model_parameters.dropout_input = 0.0

        # Tam bagli katmanda dropout
        model_parameters.dropout_dense = 0.5

    elif model_name == 'model_custom_mlp_multi':

        # goruntu boyutu
        model_parameters.pixel_count = [4, 5]

        # ardisil pencere sayisi
        model_parameters.seq_win_count = 16

        # Elektrot yerlesimleri yukleniyor
        model_parameters.locs_2d = load_locations(withProjection=False)

        # Imge olusturma yontemi
        model_parameters.gen_image_type = 'basic'

        # Tam Bagli Katman Hucre Sayisi
        model_parameters.dense_num_unit = [512, 512, 512]

        # Giriste dropout
        model_parameters.dropout_input = 0.2

        # Tam bagli katmanda dropout
        model_parameters.dropout_dense = 0.5

    elif model_name == 'model_cnn_basic':

        # goruntu boyutu
        model_parameters.pixel_count = [4, 5]

        # ardisil pencere sayisi
        model_parameters.seq_win_count = 1

        # Elektrot yerlesimleri yukleniyor
        model_parameters.locs_2d = load_locations(withProjection=False)

        # Imge olusturma yontemi
        model_parameters.gen_image_type = 'basic'

        # CNN katman Sayisi
        model_parameters.n_layers = [3]

        # Ilk filtre boyutu
        model_parameters.n_filters_first = 32

        # Tam Bagli Katman Hucre Sayisi
        model_parameters.dense_num_unit = [128]

    elif model_name == 'model_cnn':

        # goruntu boyutu
        # [16, 16], [8, 8]
        model_parameters.pixel_count = [16, 16]

        # ardisil pencere sayisi
        model_parameters.seq_win_count = 1

        # Elektrot yerlesimleri yukleniyor
        model_parameters.locs_2d = load_locations(withProjection=True)

        # Imge olusturma yontemi
        model_parameters.gen_image_type = 'advanced'

        if model_parameters.pixel_count == [16, 16]:

            # CNN Katman Sayisi
            model_parameters.n_layers = [3, 2, 1]

            # Ilk filtre boyutu
            model_parameters.n_filters_first = 32

            # Tam Bagli Katman Hucre Sayisi
            model_parameters.dense_num_unit = [512, 512]

            # CNN icinde batch normalizasyon
            model_parameters.batch_norm_conv = False

        elif model_parameters.pixel_count == [8, 8]:

            # CNN Katman Sayisi
            model_parameters.n_layers = [2, 1]

            # Ilk filtre boyutu
            model_parameters.n_filters_first = 16

            # Tam Bagli Katman Hucre Sayisi
            model_parameters.dense_num_unit = [128, 128]

            # CNN icinde batch normalizasyon
            model_parameters.batch_norm_conv = False



    elif model_name == 'model_cnn_max':

        # goruntu boyutu
        # [16, 16], [8, 8]
        model_parameters.pixel_count = [8, 8]

        # ardisil pencere sayisi
        model_parameters.seq_win_count = 16

        # Elektrot yerlesimleri yukleniyor
        model_parameters.locs_2d = load_locations(withProjection=True)

        # Imge olusturma yontemi
        model_parameters.gen_image_type = 'advanced'

        if model_parameters.pixel_count == [8, 8]:

            # CNN Katman Sayisi
            model_parameters.n_layers = [2, 1]

            # Ilk filtre boyutu
            model_parameters.n_filters_first = 16

            # Tam Bagli Katman Hucre Sayisi
            model_parameters.dense_num_unit = [128, 128]

            # CNN icinde batch normalizasyon
            model_parameters.batch_norm_conv = False


    elif model_name == 'model_cnn_conv1d':

        # goruntu boyutu
        # [16, 16], [8, 8]
        model_parameters.pixel_count = [8, 8]

        # ardisil pencere sayisi
        model_parameters.seq_win_count = 16

        # Elektrot yerlesimleri yukleniyor
        model_parameters.locs_2d = load_locations(withProjection=True)

        # Imge olusturma yontemi
        model_parameters.gen_image_type = 'advanced'

        if model_parameters.pixel_count == [8, 8]:

            # CNN Katman Sayisi
            model_parameters.n_layers = [2, 1]

            # Ilk filtre boyutu
            model_parameters.n_filters_first = 16

            # Tam Bagli Katman Hucre Sayisi
            model_parameters.dense_num_unit = [256, 256]

            # CNN icinde batch normalizasyon
            model_parameters.batch_norm_conv = False

    elif model_name == 'model_cnn_lstm':

        # goruntu boyutu
        # [16, 16], [8, 8]
        model_parameters.pixel_count = [8, 8]

        # ardisil pencere sayisi
        model_parameters.seq_win_count = 16

        # Elektrot yerlesimleri yukleniyor
        model_parameters.locs_2d = load_locations(withProjection=True)

        # Imge olusturma yontemi
        model_parameters.gen_image_type = 'advanced'

        if model_parameters.pixel_count == [8, 8]:

            # CNN Katman Sayisi
            model_parameters.n_layers = [2, 1]

            # Ilk filtre boyutu
            model_parameters.n_filters_first = 16

            # Tam Bagli Katman Hucre Sayisi
            model_parameters.dense_num_unit = [256, 256]

            # CNN icinde batch normalizasyon
            model_parameters.batch_norm_conv = False

    elif model_name == 'model_cnn_mix':

        # goruntu boyutu
        # [16, 16], [8, 8]
        model_parameters.pixel_count = [8, 8]

        # ardisil pencere sayisi
        model_parameters.seq_win_count = 16

        # Elektrot yerlesimleri yukleniyor
        model_parameters.locs_2d = load_locations(withProjection=True)

        # Imge olusturma yontemi
        model_parameters.gen_image_type = 'advanced'

        if model_parameters.pixel_count == [8, 8]:

            # CNN Katman Sayisi
            model_parameters.n_layers = [2, 1]

            # Ilk filtre boyutu
            model_parameters.n_filters_first = 16

            # Tam Bagli Katman Hucre Sayisi
            model_parameters.dense_num_unit = [256, 256]

            # CNN icinde batch normalizasyon
            model_parameters.batch_norm_conv = False

    elif model_name == 'model_cnn_lstm_hybrid':

        # goruntu boyutu
        # [16, 16], [8, 8]
        model_parameters.pixel_count = [8, 8]

        # ardisil pencere sayisi
        model_parameters.seq_win_count = 16

        # Elektrot yerlesimleri yukleniyor
        model_parameters.locs_2d = load_locations(withProjection=True)

        # Imge olusturma yontemi
        model_parameters.gen_image_type = 'advanced'

        if model_parameters.pixel_count == [8, 8]:

            # CNN Katman Sayisi
            model_parameters.n_layers = [2, 1]

            # Ilk filtre boyutu
            model_parameters.n_filters_first = 16

            # Tam Bagli Katman Hucre Sayisi
            model_parameters.dense_num_unit = [256, 256]

            # CNN icinde batch normalizasyon
            model_parameters.batch_norm_conv = False

    elif model_name == 'model_cnn3d':

        # goruntu boyutu
        # [16, 16], [8, 8]
        model_parameters.pixel_count = [8, 8]

        # ardisil pencere sayisi
        model_parameters.seq_win_count = 16

        # Elektrot yerlesimleri yukleniyor
        model_parameters.locs_2d = load_locations(withProjection=True)

        # Imge olusturma yontemi
        model_parameters.gen_image_type = 'advanced'

        if model_parameters.pixel_count == [8, 8]:

            # CNN Katman Sayisi
            model_parameters.n_layers = [1, 2, 1]

            # Ilk filtre boyutu
            model_parameters.n_filters_first = 8

            # Tam Bagli Katman Hucre Sayisi
            model_parameters.dense_num_unit = [256, 256]

            # CNN icinde batch normalizasyon
            model_parameters.batch_norm_conv = False

            # CNN maxpool boyutu
            model_parameters.pool_size = [(2,1,1),(2,2,2),(2,2,2)]

            # CNN filter boyutu
            model_parameters.filter_size = [(3, 1, 1), (3, 3, 3), (3, 3, 3)]

            # CNN filter sayisi carpani
            model_parameters.filter_factor = 2

            # Tam bagli katmanda dropout
            model_parameters.dropout_dense = True


    return model_parameters




