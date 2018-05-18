from __future__ import print_function
import time
import numpy as np
import theano
import lasagne
import theano.tensor as T
import matplotlib.pyplot as plt

from eegpredict.functions.network_models import network_convpool_cnn, network_convpool_cnn_max, network_convpool_conv1d, \
    network_convpool_lstm, network_convpool_mix, network_custom_mlp, \
    network_custom_mlp_multi, network_convpool_lstm_hybrid, network_convpool_cnn3d, network_convpool_cnn_basic
from eegpredict.functions.utils import reformatInput, standardData, moving_average, EmptyClass, calc_mean, \
    print_dashed_line, swapDataAxes, calc_avg_val_loss

from eegpredict.functions.plot_data import plot_only_prediction

from lasagne import regularization

from sklearn import svm
from sklearn.metrics import accuracy_score, log_loss, hinge_loss


def sigmoid_binary_crossentropy(predictions, targets, treshold=0.5, alfa=6, beta=0.602):
    """
    Compute the crossentropy of binary random variables.

    Output and target are each expectations of binary random
    variables; target may be exactly 0 or 1 but output must
    lie strictly between 0 and 1.

    Notes
    -----
    We could use the x log y op to support output=0 and output=1.
    The gradient would still be undefined though.

    We do not sum, crossentropy is computed by component.
    TODO : Rewrite as a scalar, and then broadcast to tensor.

    """
    predictions, targets = lasagne.objectives.align_targets(predictions, targets)
    return targets * (beta/(1+T.exp(-alfa*(treshold-predictions)))) \
           + (1.0 - targets) * (beta/(1+T.exp(-alfa*(predictions-treshold))))


def sigmoid_binary_crossentropy_mix(predictions, targets, treshold=0.5, alfa=10, beta=1):
    """
    Compute the crossentropy of binary random variables.

    Output and target are each expectations of binary random
    variables; target may be exactly 0 or 1 but output must
    lie strictly between 0 and 1.

    Notes
    -----
    We could use the x log y op to support output=0 and output=1.
    The gradient would still be undefined though.

    We do not sum, crossentropy is computed by component.
    TODO : Rewrite as a scalar, and then broadcast to tensor.

    """
    predictions, targets = lasagne.objectives.align_targets(predictions, targets)
    return targets * (beta/(1+T.exp(-alfa*(treshold-predictions)))) \
           - (1.0 - targets) * T.log(1.0 - predictions)

def binary_crossentropy(predictions, targets):
    """
    Compute the crossentropy of binary random variables.

    Output and target are each expectations of binary random
    variables; target may be exactly 0 or 1 but output must
    lie strictly between 0 and 1.

    Notes
    -----
    We could use the x log y op to support output=0 and output=1.
    The gradient would still be undefined though.

    We do not sum, crossentropy is computed by component.
    TODO : Rewrite as a scalar, and then broadcast to tensor.

    """
    predictions, targets = lasagne.objectives.align_targets(predictions, targets)
    return -(targets * T.log(predictions) + (1.0 - targets)  * T.log(1.0 - predictions))

def fbeta_bce_loss(predictions, targets, beta = 2):
    predictions, targets = lasagne.objectives.align_targets(predictions, targets)
    beta_sq = beta ** 2
    tp_loss = T.sum(targets * (1 + T.log(predictions)))
    fp_loss = T.sum((1 - targets) * -T.log(1.0 - predictions))

    return -(1 + beta_sq) * tp_loss / ((beta_sq * T.sum(targets)) + tp_loss + fp_loss)



class EpochClass(object):
    def __init__(self):
        self.duration = []
        self.av_train_err = []
        self.av_val_err = []
        self.av_val_acc = []
        self.av_val_avg_err = []
        self.net_param_val = []

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, model_name='model_cnn'):
    """
    Iterates over the samples returing batches of size batchsize.
    :param inputs: input data array. It should be a 4D numpy array for images [n_samples, n_colors, W, H] and 5D numpy
                    array if working with sequence of images [n_timewindows, n_samples, n_colors, W, H].
    :param targets: vector of target labels.
    :param batchsize: Batch size
    :param shuffle: Flag whether to shuffle the samples before iterating or not.
    :return: images and labels for a batch
    """
    if inputs.ndim == 4 or model_name == 'model_cnn3d' or model_name =='model_cnn3d_basic' or model_name =='model_cnn_temp':
        input_len = inputs.shape[0]
    elif inputs.ndim == 5:
        input_len = inputs.shape[1]
    assert input_len == len(targets)
    if shuffle:
        indices = np.arange(input_len)
        np.random.shuffle(indices)
    for start_idx in range(0, input_len, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if inputs.ndim == 4 or model_name == 'model_cnn3d' or model_name =='model_cnn3d_basic' or model_name =='model_cnn_temp':
            yield inputs[excerpt], targets[excerpt]
        elif inputs.ndim == 5:
            yield inputs[:, excerpt], targets[excerpt]

def iterate_minibatches_online_weights(inputs, targets, weights, batchsize, shuffle=False, model_name='model_cnn'):
    """
    Iterates over the samples returing batches of size batchsize.
    :param inputs: input data array. It should be a 4D numpy array for images [n_samples, n_colors, W, H] and 5D numpy
                    array if working with sequence of images [n_timewindows, n_samples, n_colors, W, H].
    :param targets: vector of target labels.
    :param batchsize: Batch size
    :param shuffle: Flag whether to shuffle the samples before iterating or not.
    :return: images and labels for a batch
    """
    if inputs.ndim == 4 or model_name == 'model_cnn3d' or model_name =='model_cnn3d_basic' or model_name =='model_cnn_temp':
        input_len = inputs.shape[0]
    elif inputs.ndim == 5:
        input_len = inputs.shape[1]
    assert input_len == len(targets)
    if shuffle:
        indices = np.arange(input_len)
        np.random.shuffle(indices)
    for start_idx in range(0, input_len, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if inputs.ndim == 4 or model_name == 'model_cnn3d' or model_name =='model_cnn3d_basic' or model_name =='model_cnn_temp':
            yield inputs[excerpt], targets[excerpt], weights[excerpt]
        elif inputs.ndim == 5:
            yield inputs[:, excerpt], targets[excerpt], weights[excerpt]


def test_with_iterate_minibatches(test_fn, x_test, y_test, batch_size, model_name='model_cnn'):
    test_err = 0
    test_acc = 0
    test_batches = 0
    predict = []
    for batch in iterate_minibatches(x_test, y_test, batch_size, shuffle=False, model_name=model_name):
        inputs, targets = batch
        err, acc, pred = test_fn(inputs, targets)
        test_err += err
        test_acc += acc
        predict = pred if predict == [] else np.vstack((predict,pred))
        test_batches += 1
    test_err = test_err / test_batches
    test_acc = test_acc / test_batches
    return test_acc, test_err, predict


def train_with_iterate_minibatches(train_fn, x_train, y_train, batch_size, model_name='model_cnn'):
    train_err = 0
    train_acc = 0
    train_batches = 0
    for batch in iterate_minibatches(x_train, y_train, batch_size, shuffle=False, model_name=model_name):
        inputs, targets= batch
        err, acc = train_fn(inputs, targets)
        train_err += err
        train_acc += acc
        train_batches += 1
    train_err = train_err / train_batches
    train_acc = train_acc / train_batches
    return train_err, train_acc

def train_with_iterate_minibatches_online_weights(train_fn, x_train, y_train, batch_size, model_name='model_cnn', train_weights=[]):
    train_err = 0
    train_acc = 0
    train_batches = 0
    for batch in iterate_minibatches_online_weights(x_train, y_train, train_weights, batch_size, shuffle=False, model_name=model_name):
        inputs, targets, weights = batch
        err, acc = train_fn(inputs, targets, weights)
        train_err += err
        train_acc += acc
        train_batches += 1
    train_err = train_err / train_batches
    train_acc = train_acc / train_batches
    return train_err, train_acc


def create_neural_network(model_params, imsize=16, n_colors=14, balanced_weights=True, class_weights=[1, 1],
                          seq_win_count=5, l1=0.0, l2=0.0, under_sample_ratio=1, learn_rate=0.001,
                          online_weights= False):
    # Number of class unit
    num_class_unit = 1

    # Giris ve cikis parametreleri icin Theano degiskenleri hazirlaniyor
    input_var = T.TensorType('floatX', ((False,) * 5))()
    target_var = T.ivector('targets')
    train_weights = T.dvector('train_weights')


    # Model olusturuluyor
    if model_params.model_name == 'model_custom_mlp':
        input_var = T.tensor4('inputs')
        network = network_custom_mlp(input_var,
                                     num_class_unit,
                                     n_colors=n_colors,
                                     imsize=imsize,
                                     width=model_params.dense_num_unit,
                                     drop_input=model_params.dropout_input,
                                     drop_hidden=model_params.dropout_dense)

    elif model_params.model_name == 'model_custom_mlp_multi':
        network = network_custom_mlp_multi(input_var,
                                           num_class_unit,
                                           n_colors=n_colors,
                                           imsize=imsize,
                                           width=model_params.dense_num_unit,
                                           drop_input=model_params.dropout_input,
                                           drop_hidden=model_params.dropout_dense,
                                           n_timewin=seq_win_count)

    elif model_params.model_name == 'model_cnn':
        input_var = T.tensor4('inputs')
        network = network_convpool_cnn(input_var,
                                       num_class_unit,
                                       n_colors=n_colors,
                                       imsize=imsize,
                                       n_layers=model_params.n_layers,
                                       n_filters_first=model_params.n_filters_first,
                                       dense_num_unit= model_params.dense_num_unit,
                                       batch_norm_conv=model_params.batch_norm_conv)

    elif model_params.model_name == 'model_cnn_basic':
        input_var = T.tensor4('inputs')
        network = network_convpool_cnn_basic(input_var,
                                             num_class_unit,
                                             n_colors=n_colors,
                                             imsize=imsize,
                                             n_layers=model_params.n_layers,
                                             n_filters_first=model_params.n_filters_first,
                                             dense_num_unit = model_params.dense_num_unit,
                                             batch_norm_conv = model_params.batch_norm_conv)

    elif model_params.model_name == 'model_cnn_max':
        network = network_convpool_cnn_max(input_var,
                                           num_class_unit,
                                           n_colors=n_colors,
                                           imsize=imsize,
                                           n_timewin=seq_win_count,
                                           n_layers=model_params.n_layers,
                                           n_filters_first=model_params.n_filters_first,
                                           dense_num_unit=model_params.dense_num_unit,
                                           batch_norm_conv=model_params.batch_norm_conv)

    elif model_params.model_name == 'model_cnn_conv1d':
        network = network_convpool_conv1d(input_var,
                                          num_class_unit,
                                          n_colors=n_colors,
                                          imsize=imsize,
                                          n_timewin=seq_win_count,
                                          n_layers=model_params.n_layers,
                                          n_filters_first=model_params.n_filters_first,
                                          dense_num_unit=model_params.dense_num_unit,
                                          batch_norm_conv=model_params.batch_norm_conv)

    elif model_params.model_name == 'model_cnn_lstm':
        network = network_convpool_lstm(input_var,
                                        num_class_unit,
                                        n_colors=n_colors,
                                        imsize=imsize,
                                        n_timewin=seq_win_count,
                                        n_layers=model_params.n_layers,
                                        n_filters_first=model_params.n_filters_first,
                                        dense_num_unit=model_params.dense_num_unit,
                                        batch_norm_conv=model_params.batch_norm_conv)

    elif model_params.model_name == 'model_cnn_mix':
        network = network_convpool_mix(input_var,
                                       num_class_unit,
                                       n_colors=n_colors,
                                       imsize=imsize,
                                       n_timewin=seq_win_count,
                                       n_layers=model_params.n_layers,
                                       n_filters_first=model_params.n_filters_first,
                                       dense_num_unit=model_params.dense_num_unit,
                                       batch_norm_conv=model_params.batch_norm_conv)

    elif model_params.model_name == 'model_cnn_lstm_hybrid':
        network = network_convpool_lstm_hybrid(input_var,
                                               num_class_unit,
                                               n_colors=n_colors,
                                               imsize=imsize,
                                               n_timewin=seq_win_count,
                                               n_layers=model_params.n_layers,
                                               n_filters_first = model_params.n_filters_first,
                                               dense_num_unit=model_params.dense_num_unit,
                                               batch_norm_conv=model_params.batch_norm_conv)

    elif model_params.model_name == 'model_cnn3d':
        network = network_convpool_cnn3d(input_var,
                                         num_class_unit,
                                         n_colors=n_colors,
                                         n_timewin=seq_win_count,
                                         imsize=imsize,
                                         n_layers=model_params.n_layers,
                                         n_filters_first=model_params.n_filters_first,
                                         dense_num_unit=model_params.dense_num_unit,
                                         batch_norm_conv=model_params.batch_norm_conv,
                                         pool_size=model_params.pool_size,
                                         filter_size=model_params.filter_size,
                                         filter_factor=model_params.filter_factor,
                                         dropout_dense=model_params.dropout_dense)
    else:
        raise ValueError("Model not supported")

    # Train icin loss fonksiyonu tanimlaniyor
    prediction = lasagne.layers.get_output(network)
    if num_class_unit == 1:
        #loss_function = lasagne.objectives.binary_crossentropy(prediction, target_var)
        loss_function = binary_crossentropy(prediction, target_var)
        #loss_function = fbeta_bce_loss(prediction, target_var)

        train_acc = lasagne.objectives.binary_accuracy(prediction, target_var, threshold=0.5)
        train_acc = T.mean(train_acc, dtype=theano.config.floatX)
    else:
        loss_function = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)

    # Loss fonksiyonun veriseti underSample yapilmadiysa agirlikli olarak, yapildiysa basit olarak ortalamasi aliniyor
    if balanced_weights == False:
        # ornek oranina gore agirliklandirma
        # etiketler preictal=1 interictal=0
        weights_per_label = theano.shared(lasagne.utils.floatX(class_weights))
        weights = weights_per_label[target_var]
        loss = lasagne.objectives.aggregate(loss_function, weights=weights)
    elif under_sample_ratio != 1:
        weights_per_label = theano.shared(lasagne.utils.floatX([under_sample_ratio, 1]))
        weights = weights_per_label[target_var]
        loss = lasagne.objectives.aggregate(loss_function, weights=weights)
    elif online_weights:
        loss = lasagne.objectives.aggregate(loss_function, weights=train_weights)
    else:
        loss = loss_function.mean()

    # regularization
    if l1:
        loss += regularization.regularize_network_params(network, regularization.l1) * l1
    if l2:
        loss += regularization.regularize_network_params(network, regularization.l2) * l2

    # Parametreleri update edecek foknsiyon
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learn_rate)

    # Validation ve test icin loss fonksiyonu tanimlaniyor
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    if num_class_unit == 1:
        #test_loss = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
        test_loss = binary_crossentropy(test_prediction, target_var)
        #test_loss = fbeta_bce_loss(test_prediction, target_var)
        test_loss = test_loss.mean()
        test_acc = lasagne.objectives.binary_accuracy(test_prediction, target_var, threshold=0.5)
        test_acc = T.mean(test_acc, dtype=theano.config.floatX)
    else:
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    # regularization
    # validation sirasinda regularization olmamali
    # if l1:
    #     test_loss += regularization.regularize_layer_params(network, regularization.l1) * l1
    # if l2:
    #     test_loss += regularization.regularize_network_params(network, regularization.l2) * l2

    # Train fonksiyonu compile ediliyor
    if online_weights:
        train_fn = theano.function([input_var, target_var,train_weights], [loss, train_acc], updates=updates)
    else:
        train_fn = theano.function([input_var, target_var], [loss, train_acc], updates=updates)
    # Validation fonksiyonu compile ediliyor
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])
    # Test fonksiyonu compile ediliyor
    test_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])

    return network, train_fn, val_fn, test_fn


def fit_neural_network(model, x_train, y_train, x_val, y_val, train_weights=[]):

    epoch_params = EpochClass()

    # her epoch dongusunde model parametreleri restore ediliyor
    lasagne.layers.set_all_param_values(model.model, model.network_param_values)



    print("\n\nepoch\t\ttrain loss\t\tvalid loss\t\terror neg\t\terror pos\t\ttrain/val\t\ttrain acc\t\tvalid acc\t\tvalid avg err\t\tdur")
    print("-------\t\t------------\t------------\t------------\t------------\t-----------\t\t-----------\t\t-----------\t\t-----------\t\t------")
    # num_epoch sayisinca iterasyon
    for epoch in range(model.train_params.num_epoch):
        start_time = time.time()

        # Her epoch'ta train data tumuyle train ediliyor
        if model.train_params.online_weights:
            av_train_err, av_train_acc = train_with_iterate_minibatches_online_weights(model.train_fn, x_train, y_train,
                                                                                       model.train_params.batch_size,
                                                                                       model_name=model.model_params.model_name,
                                                                                       train_weights=train_weights)
        else:
            av_train_err, av_train_acc = train_with_iterate_minibatches(model.train_fn, x_train, y_train,
                                                                        model.train_params.batch_size,
                                                                        model_name=model.model_params.model_name)

        # Validation datanin tumu test ediliyor
        val_err_pre, val_acc_pre, val_pred_pre = model.val_fn(x_val[0], y_val[0])
        val_err_int, val_acc_int, val_pred_int = model.val_fn(x_val[1], y_val[1])

        # Ortalama validation hatasi ve dogrulugu
        av_val_acc = calc_mean(val_acc_pre, val_acc_int, y_val[0].shape[0], y_val[1].shape[0], 'basic')
        av_val_err = calc_mean(val_err_pre, val_err_int, y_val[0].shape[0], y_val[1].shape[0], 'basic')
        av_val_avg_err = calc_avg_val_loss(val_pred_pre,val_pred_int,valid_avg_size=30)
        # pred_pre_max, pred_int_max = calc_pred_max(val_pred_pre,val_pred_int,valid_avg_size=train_params.valid_avg_size)
        # min_val_diff = pred_pre_max - pred_int_max


        # train ve validation icin hata ve dogruluk degeri ve ag parametreleri saklaniyor
        epoch_params.duration.append(time.time() - start_time)
        epoch_params.av_train_err.append(av_train_err)
        epoch_params.av_val_err.append(av_val_err)
        epoch_params.av_val_acc.append(av_val_acc)
        epoch_params.av_val_avg_err.append(av_val_avg_err)
        epoch_params.net_param_val.append(lasagne.layers.get_all_param_values(model.model))

        print("\t{}\t\t{:.6f}\t\t{:.6f}\t\t{:.6f}\t\t{:.6f}\t\t{:.6f}\t\t{:.2f} %\t\t\t{:.2f} %\t\t\t{:.6f}\t\t\t{:.3f}s".format(epoch+1,
                                                                                    av_train_err, av_val_err,
                                                                                    float(val_err_int),
                                                                                    float(val_err_pre),
                                                                                    av_train_err/av_val_err,
                                                                                    av_train_acc * 100,
                                                                                    av_val_acc * 100,
                                                                                    float(av_val_avg_err),
                                                                                    time.time() - start_time))
    print("")

    best_valid_idx_count = 1
    if best_valid_idx_count == 1:
        # en iyi sonucu veren indeks bulunuyor
        best_idx = np.argmin(epoch_params.av_val_err)
        # en iyi sonuc icin parametreler aga yukleniyor
        lasagne.layers.set_all_param_values(model.model, epoch_params.net_param_val[best_idx])
        epoch_params.net_param_val = epoch_params.net_param_val[best_idx]
    else :
        # en iyi sonucu veren indeksler bulunuyor
        best_idx = np.argsort(epoch_params.av_val_err)[:best_valid_idx_count]
        net_params = np.asarray(epoch_params.net_param_val[best_idx[0]])/best_valid_idx_count
        for i in range(1,best_valid_idx_count):
            net_params = net_params + np.asarray(epoch_params.net_param_val[best_idx[i]])/best_valid_idx_count
        lasagne.layers.set_all_param_values(model.model, net_params)
        #epoch_params.net_param_val = net_params

    #best_idx = np.argmin(epoch_params.av_val_avg_err)



    # Egitilen agin agirliklari dosyaya kaydediliyor
    # np.savez(
    #     '../model/weights_{}_p{}_s{}_f{}'.format(model.model_params.model_name, dataset.pat_id, sample, f),
    #     *net_param_val[best_idx])


    return model, epoch_params


def fit_svm(model, x_train, y_train, x_val, y_val):

    # validate verileri train verilerine ekleniyor
    x_train = np.concatenate((x_train, x_val[0], x_val[1]))
    y_train = np.concatenate((y_train, y_val[0], y_val[1]))

    # randomize ediliyor
    trainIndices = range(x_train.shape[0])
    np.random.shuffle(trainIndices)
    x_train = x_train[trainIndices]
    y_train = y_train[trainIndices]

    # oznitelikler tek boyutlu hale getiriliyor
    x_train = np.reshape(x_train, (x_train.shape[0], -1))

    # svm egitiliyor
    model.model.fit(x_train, y_train)

    return model


def cross_validation(model, dataset, save_model_params=False, plot_prediction=False, pre_trained_init=False):

    train_result_all = []
    test_result_all = []
    predict_result_all = []

    # ilk dongude bir ornek cifti disarida birakilarak digerleri ile egitiliyor
    for sample in range(0, model.train_params.num_sample):

        train_result = []
        test_result = []
        predict_result = []

        # ikinci dongude N-fold cross validation yapiliyor
        #for fold in range(train_params.num_fold):
        for fold in range(0,1):

            print('Sample: {}\tFold: {}\t\t'.format(sample+1, fold+1), end='')

            # train, valid ve test verileri ayarlaniyor
            (X_train, y_train, train_weights), (X_val, y_val), (X_test, y_test) = reformatInput(dataset.images,
                                                                                 dataset.fold_pairs[sample], fold,
                                                                                 model.model_params.seq_win_count,
                                                                                 under_sample=dataset.under_sample,
                                                                                 under_sample_ratio=dataset.under_sample_ratio,
                                                                                 model_name=model.model_params.model_name)
            # veriler normalize ediliyor
            X_train, X_val, X_test = standardData(X_train, X_val, X_test, with_mean=True,
                                                  model_name=model.model_params.model_name)

            # veriler float32 olarak alinacak
            X_train = X_train.astype("float32", casting='unsafe')
            X_val[0] = X_val[0].astype("float32", casting='unsafe')  # preictal valid
            X_val[1] = X_val[1].astype("float32", casting='unsafe')  # interictal valid
            X_test[0] = X_test[0].astype("float32", casting='unsafe')  # preictal test
            X_test[1] = X_test[1].astype("float32", casting='unsafe')  # interictal test

            # model egitiliyor
            if model.model_type == "neural":

                if pre_trained_init:
                    net_param_val = np.load('../model/weights_{}_pre_train_params.npz'.format(model.model_params.model_name))['arr_0.npy']
                    model.network_param_values = net_param_val

                model, epoch_params = fit_neural_network(model, X_train, y_train, X_val, y_val, train_weights)

                # for i in range(len(epoch_params.net_param_val)):
                #     lasagne.layers.set_all_param_values(model.model, epoch_params.net_param_val[i])
                #     test_acc_pre, test_err_pre, predict_pre = test_with_iterate_minibatches(model.test_fn, X_test[0],
                #                                                                             y_test[0],
                #                                                                             train_params.batch_size,
                #                                                                             model.model_params.model_name)
                #     test_acc_int, test_err_int, predict_int = test_with_iterate_minibatches(model.test_fn, X_test[1],
                #                                                                             y_test[1],
                #                                                                             train_params.batch_size,
                #                                                                             model.model_params.model_name)
                #     plot_only_prediction(predict_int, predict_pre)

                if save_model_params:
                    np.savez(
                        '../model/weights_{}_p{}_s{}_f{}'.format(model.model_params.model_name, dataset.pat_id, sample, fold),
                        epoch_params.net_param_val)
                epoch_params.net_param_val = []
            elif model.model_type == "svm":
                model = fit_svm(model, X_train, y_train, X_val, y_val)
                epoch_params = []


            # model test ediliyor
            if model.model_type == "neural":
                test_acc_pre, test_err_pre, predict_pre = test_with_iterate_minibatches(model.test_fn, X_test[0],
                                                                                        y_test[0],
                                                                                        model.train_params.batch_size,
                                                                                        model.model_params.model_name)
                test_acc_int, test_err_int, predict_int = test_with_iterate_minibatches(model.test_fn, X_test[1],
                                                                                        y_test[1],
                                                                                        model.train_params.batch_size,
                                                                                        model.model_params.model_name)
                # ortalama hata ve dogruluk icin pozitif ve negatif ornekler icin ortalama aliniyor
                av_test_acc = calc_mean(test_acc_pre, test_acc_int, y_test[0].shape[0], y_test[1].shape[0], 'basic')
                av_test_err = calc_mean(test_err_pre, test_err_int, y_test[0].shape[0], y_test[1].shape[0], 'basic')

            elif model.model_type == "svm":
                # oznitelikler tek boyutlu hale getiriliyor
                X_test[0] = np.reshape(X_test[0], (X_test[0].shape[0], -1))
                X_test[1] = np.reshape(X_test[1], (X_test[1].shape[0], -1))

                predict_pre = model.model.predict(X_test[0])
                predict_int = model.model.predict(X_test[1])

                test_acc_pre = accuracy_score(y_test[0], predict_pre)
                test_acc_int = accuracy_score(y_test[1], predict_int)

                if 0 in predict_pre:
                    test_err_pre = log_loss(predict_pre, y_test[0])
                else:
                    test_err_pre = 0
                test_err_int = log_loss(predict_int, y_test[1])

                # ortalama dogruluk icin pozitif ve negatif ornekler icin ortalama aliniyor
                av_test_acc = calc_mean(test_acc_pre, test_acc_int, y_test[0].shape[0], y_test[1].shape[0], 'basic')
                av_test_err = calc_mean(test_err_pre, test_err_int, y_test[0].shape[0], y_test[1].shape[0], 'basic')


            # sonuclar geri donuluyor
            train_result.append(epoch_params)
            test_result.append([av_test_err, av_test_acc])
            predict_result.append([predict_pre, predict_int])
            print("test loss: {:.6f}\t\ttest accuracy: {:.2f} %".format(av_test_err, av_test_acc * 100))
            print_dashed_line()

            # fold icin prediction sonuclarini cizdir
            if plot_prediction:
                plot_only_prediction(predict_int,predict_pre)

        print_dashed_line()
        train_result_all.append(train_result)
        test_result_all.append(test_result)
        predict_result_all.append(predict_result)
    return train_result_all, test_result_all, predict_result_all


def create_model(model_params, feat_params, train_params, dataset):
    mdl = EmptyClass()
    mdl.model_params = model_params
    mdl.train_params = train_params
    if model_params.model_name == 'model_svm':
        mdl.model_type = "svm"
        mdl.model = svm.SVC(kernel=model_params.kernel, class_weight='balanced', probability=False)
    else:
        mdl.model_type = "neural"
        mdl.model, mdl.train_fn, mdl.val_fn, mdl.test_fn = create_neural_network(model_params,
                                                                                 imsize=model_params.pixel_count,
                                                                                 n_colors=feat_params.feature_count,
                                                                                 balanced_weights=dataset.under_sample,
                                                                                 class_weights=dataset.class_weights,
                                                                                 seq_win_count=model_params.seq_win_count,
                                                                                 l2=train_params.l2,
                                                                                 l1=train_params.l1,
                                                                                 under_sample_ratio=dataset.under_sample_ratio,
                                                                                 learn_rate=train_params.learn_rate,
                                                                                 online_weights=train_params.online_weights)
    return mdl


def pre_train_model(model, dataset):

    print('Model on egitiliyor...')
    print_dashed_line()

    if model.model_type == "neural":
        # network'un ilklendirilmis parametreleri restore edilmek uzere kaydedildi
        model.network_param_values = lasagne.layers.get_all_param_values(model.model)

    pre_label = 1
    int_label = 0

    # datasette nan ifadeler varsa numerik degerlerle degistirilecek
    data_pre = np.nan_to_num(dataset.images[0])
    data_int = np.nan_to_num(dataset.images[1])

    # verilerin indeksleri karistiriliyor
    data_pre_idx = range(len(data_pre))
    data_int_idx = range(len(data_int))
    np.random.shuffle(data_pre_idx)
    np.random.shuffle(data_int_idx)

    # veriler train ve valid olarak ayriliyor

    X_val= [data_pre[data_pre_idx[0:len(data_pre)/5]], data_int[data_int_idx[0:len(data_int)/5]]]
    validLabels = [np.full(len(data_pre) / 5, pre_label).astype(np.int32),
                   np.full(len(data_int) / 5, int_label).astype(np.int32)]
    y_val = validLabels

    X_train = np.concatenate((data_pre[data_pre_idx[len(data_pre) / 5:]], data_int[data_int_idx[len(data_int) / 5:]]),
                           axis=0)
    trainLabels = np.concatenate(
        (np.full(len(data_pre)-(len(data_pre) / 5), pre_label).astype(np.int32),
         np.full(len(data_int)-(len(data_int) / 5), int_label).astype(np.int32)), axis=0)
    trainIndices = range(len(X_train))
    np.random.shuffle(trainIndices)
    X_train = X_train[trainIndices]
    y_train = trainLabels[trainIndices].astype(np.int32)

    # dummy X_test
    #X_test = X_val[:]

    # veriler normalize ediliyor
    #X_train, X_val, _ = standardData(X_train, X_val, X_test, with_mean=False,
    #                                      model_name=model.model_params.model_name)

    # veriler float32 olarak alinacak
    X_train = X_train.astype("float32", casting='unsafe')
    X_val[0] = X_val[0].astype("float32", casting='unsafe')  # preictal valid
    X_val[1] = X_val[1].astype("float32", casting='unsafe')  # interictal valid

    # model egitiliyor
    if model.model_type == "neural":

        model, epoch_params = fit_neural_network(model, X_train, y_train, X_val, y_val)

        if model.train_params.save_model_params:
            np.savez(
                '../model/weights_{}_pre_train_params'.format(model.model_params.model_name),
                epoch_params.net_param_val)
        epoch_params.net_param_val = []
    elif model.model_type == "svm":
        model = fit_svm(model, X_train, y_train, X_val, y_val)
        epoch_params = []

    return


def train_model(model, dataset):

    print('Model egitiliyor...')
    print_dashed_line()


    if model.model_type == "neural":
        # network'un ilklendirilmis parametreleri restore edilmek uzere kaydedildi
        model.network_param_values = lasagne.layers.get_all_param_values(model.model)

    # datasetteki nobet sayisi
    model.train_params.num_sample = np.asarray(dataset.fold_pairs).shape[0]

    # fold sayisi
    model.train_params.num_fold = len(np.asarray(dataset.fold_pairs)[0][0])

    train_result, test_result, predict_result = cross_validation(model, dataset)

    return train_result, test_result, predict_result