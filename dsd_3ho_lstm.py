import sys

import keras.regularizers
import tensorflow as tf
import tempfile
import os
import zipfile
import tensorflow_model_optimization as tfmot
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import math
from sklearn.metrics import precision_recall_curve

import pickle


def create_sparsity_masks(model, sparsity):
    weights_list = model.get_weights()
    masks = []
    for weights in weights_list:
        if len(weights.shape) > 1:
            weights_abs = weights
            masks.append((weights_abs > np.percentile(weights_abs, sparsity)) * 1.)
    return masks


class Regularization(tf.keras.regularizers.Regularizer):
    def __init__(self, a, target):
        self.a = a
        self.target = target

    def __call__(self, weights):
        mask = self.get_mask(weights, self.target)
        return 0.5 * self.a * tf.reduce_sum(tf.pow(weights * (1 - mask), 2))

    def get_mask(self, weights, target):
        mask = tf.cast(tf.greater(weights, target), dtype=tf.float32)
        return mask

    def get_config(self):
        return {'a': self.a, 'target': self.target}


class Sparse(tf.keras.constraints.Constraint):

    def __init__(self, mask):
        self.mask = tf.keras.backend.cast_to_floatx(mask)

    def __call__(self, x):
        return self.mask * x

    def get_config(self):
        return {'mask': self.mask}


def create_model():
    modelouter = tf.keras.Sequential()
    modelouter.add(tf.keras.layers.LSTM(32, input_shape=(None, 33), return_sequences=True))
    modelouter.add(tf.keras.layers.Dropout(0.1))
    modelouter.add(tf.keras.layers.LSTM(32, return_sequences=True))
    modelouter.add(tf.keras.layers.Dropout(0.1))
    modelouter.add(tf.keras.layers.LSTM(32, return_sequences=False))
    modelouter.add(tf.keras.layers.Dropout(0.1))
    modelouter.add(tf.keras.layers.Dense(1))
    modelouter.add(tf.keras.layers.Activation("sigmoid"))
    modelouter.summary()
    return modelouter


a = 0.001  # Hệ số regularization
target = 0.5  # Ngưỡng cho lớp Regularization


def create_sparse_model(model, sparsity):
    masks = create_sparsity_masks(model, sparsity)
    modelouter = tf.keras.Sequential()
    modelouter.add(tf.keras.layers.LSTM(32, input_shape=(None, 33), return_sequences=True))
    modelouter.add(tf.keras.layers.Dropout(0.1))
    modelouter.add(tf.keras.layers.LSTM(32, return_sequences=True, kernel_constraint=Sparse(masks[2]),
                                        kernel_regularizer=Regularization(a, target)))
    modelouter.add(tf.keras.layers.Dropout(0.1))
    modelouter.add(tf.keras.layers.LSTM(32, return_sequences=False, kernel_constraint=Sparse(masks[4]),
                                        kernel_regularizer=Regularization(a, target)))
    modelouter.add(tf.keras.layers.Dropout(0.1))
    modelouter.add(tf.keras.layers.Dense(1, kernel_constraint=Sparse(masks[6]),
                                         kernel_regularizer=Regularization(a, target)))
    modelouter.add(tf.keras.layers.Activation("sigmoid"))
    modelouter.summary()
    return modelouter


def create_re_dense_model():
    modelouter = tf.keras.Sequential()
    modelouter.add(tf.keras.layers.LSTM(32, input_shape=(None, 33), return_sequences=True))
    modelouter.add(tf.keras.layers.Dropout(0.1))
    modelouter.add(tf.keras.layers.LSTM(32, return_sequences=True))
    modelouter.add(tf.keras.layers.Dropout(0.1))
    modelouter.add(tf.keras.layers.LSTM(32, return_sequences=False))
    modelouter.add(tf.keras.layers.Dropout(0.1))
    modelouter.add(tf.keras.layers.Dense(1))
    modelouter.add(tf.keras.layers.Activation("sigmoid"))
    modelouter.summary()
    return modelouter


def load_data(filePath):
    dataset = pd.read_csv(filePath, low_memory=False)
    le = preprocessing.LabelEncoder()
    dataset['proto'] = le.fit_transform(dataset['proto'].astype(str))
    dataset['state'] = le.fit_transform(dataset['state'].astype(str))
    dataset['service'] = le.fit_transform(dataset['service'].astype(str))
    dataset = dataset.drop(
        ['srcip', 'sport', 'dstip', 'sbytes', 'dbytes', 'res_bdy_len', 'Sintpkt', 'Dintpkt', 'is_sm_ips_ports',
         'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_dst_ltm', 'ct_src_dport_ltm', 'attack_cat', 'proto_le',
         'state_le', 'service_le'], axis=1)
    X = dataset.iloc[:, 0:33]
    y = dataset.iloc[:, 33]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    return X, y


def load_data_validation(filePath):
    dataset = pd.read_csv(filePath, low_memory=False)
    le = preprocessing.LabelEncoder()
    dataset['proto'] = le.fit_transform(dataset['proto'].astype(str))
    dataset['state'] = le.fit_transform(dataset['state'].astype(str))
    dataset['service'] = le.fit_transform(dataset['service'].astype(str))
    dataset = dataset.drop(
        ['srcip', 'sport', 'dstip', 'sbytes', 'dbytes', 'res_bdy_len', 'Sintpkt', 'Dintpkt', 'is_sm_ips_ports',
         'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_dst_ltm', 'ct_src_dport_ltm', 'attack_cat', 'proto_le',
         'state_le', 'service_le'], axis=1)
    X = dataset.iloc[:, 0:33]
    y = dataset.iloc[:, 33]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    return X, y


def load_data_test(filePath):
    dataset = pd.read_csv(filePath, low_memory=False)
    le = preprocessing.LabelEncoder()
    dataset['proto'] = le.fit_transform(dataset['proto'].astype(str))
    dataset['state'] = le.fit_transform(dataset['state'].astype(str))
    dataset['service'] = le.fit_transform(dataset['service'].astype(str))
    dataset = dataset.drop(
        ['srcip', 'sport', 'dstip', 'sbytes', 'dbytes', 'res_bdy_len', 'Sintpkt', 'Dintpkt', 'is_sm_ips_ports',
         'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_dst_ltm', 'ct_src_dport_ltm', 'attack_cat', 'proto_le',
         'state_le', 'service_le'], axis=1)
    X = dataset.iloc[:, 0:33]
    y = dataset.iloc[:, 33]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X, y


def train_data(batch_t, epoch_t, sparsity):
    X_train, y_train = load_data('Data/unsw_train.csv')
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test, y_test = load_data_test('Data/unsw_test.csv')
    X_validation, y_validation = load_data_validation('Data/unsw_validation.csv')
    X_validation = np.reshape(X_validation, (X_validation.shape[0], 1, X_validation.shape[1]))

    model = create_model()
    csv_logger = tf.keras.callbacks.CSVLogger("model_dsd_op/train/train_analysis.csv", separator=',', append=False)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=15,
                                          restore_best_weights=True)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="model_dsd_op/train/train_model.hdf5",
                                                      verbose=1,
                                                      save_best_only=True, monitor='val_accuracy', mode='auto')
    checkpointer_2 = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_dsd_op/train/checkpoint-{epoch:02d}.hdf5",
        verbose=1,
        save_best_only=True, monitor='val_accuracy', mode='auto')
    learning_rate = 0.1
    momentum = 0.9
    sgd_optimizer_1 = tf.keras.optimizers.SGD(learning_rate, momentum)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=sgd_optimizer_1,
                  metrics=['accuracy'])
    history = model.fit(
        X_train,
        y_train,
        epochs=epoch_t, validation_data=(X_validation, y_validation),
        callbacks=[checkpointer, csv_logger, es, checkpointer_2]
    )
    # model = tf.keras.models.load_model( "model_dsd/train/train_model_" + str(epoch_t) + "_" + str(batch_t) + "_save")
    baseline_model_accuracy_loss, baseline_model_accuracy = model.evaluate(
        X_test, y_test, verbose=0)
    model.save_weights("model_dsd_op/train/train_model_weights_" + str(epoch_t) + "_" + str(batch_t) + ".hdf5")
    tf.keras.models.save_model(model, "model_dsd_op/train/train_model_" + str(epoch_t) + "_" + str(batch_t) + "_save")
    with open("model_dsd_op/train/train_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    print('Baseline test loss:', baseline_model_accuracy_loss)
    print('Baseline test accuracy:', baseline_model_accuracy)

    sparse_model = create_sparse_model(model, sparsity)

    learning_rate = 0.01
    momentum = 0.9
    sgd_optimizer_2 = tf.keras.optimizers.SGD(learning_rate, momentum)
    checkpointer_3 = tf.keras.callbacks.ModelCheckpoint(filepath="model_dsd_op/sparse/sparse_model.hdf5",
                                                        verbose=1,
                                                        save_best_only=True, monitor='val_accuracy', mode='auto')
    checkpointer_4 = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_dsd_op/sparse/sparse_checkpoint-{epoch:02d}.hdf5",
        verbose=1,
        save_best_only=True, monitor='val_accuracy', mode='auto')
    csv_logger_2 = tf.keras.callbacks.CSVLogger("model_dsd_op/sparse/sparse_analysis.csv", separator=',', append=False)
    sparse_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=sgd_optimizer_2,
                         metrics=['accuracy'])
    sparse_model.summary()
    sparse_model.set_weights(model.get_weights())
    history_2 = sparse_model.fit(
        X_train,
        y_train,
        epochs=epoch_t, validation_data=(X_validation, y_validation),
        callbacks=[checkpointer_3, csv_logger_2, es, checkpointer_4]
    )
    sparse_model_accuracy_loss, sparse_model_accuracy = sparse_model.evaluate(
        X_test, y_test, verbose=0)
    print('Baseline test loss:', sparse_model_accuracy_loss)
    print('Baseline test accuracy:', sparse_model_accuracy)

    sparse_model.save_weights("model_dsd_op/sparse/sparse_model_weights_" + str(epoch_t) + "_" + str(batch_t) + ".hdf5")
    tf.keras.models.save_model(sparse_model,
                               "model_dsd_op/sparse/sparse_model_" + str(epoch_t) + "_" + str(batch_t) + "_save")
    with open("model_dsd_op/sparse/sparse_history.pkl", "wb") as f:
        pickle.dump(history_2.history, f)

    # sparse_model = tf.keras.models.load_model( "model_dsd/sparse/sparse_model_" + str(epoch_t) + "_" + str(batch_t) + "_save")
    # sparse_model.summary()
    model_re_dense = create_re_dense_model()
    model_re_dense.load_weights(
        "model_dsd_op/sparse/sparse_model_weights_" + str(epoch_t) + "_" + str(batch_t) + ".hdf5")
    learning_rate = 0.001
    momentum = 0.9
    sgd_optimizer_3 = tf.keras.optimizers.SGD(learning_rate, momentum)
    checkpointer_5 = tf.keras.callbacks.ModelCheckpoint(filepath="model_dsd_op/redense/redense_model.hdf5",
                                                        verbose=1,
                                                        save_best_only=True, monitor='val_accuracy', mode='auto')
    checkpointer_6 = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_dsd_op/redense/redense_checkpoint-{epoch:02d}.hdf5",
        verbose=1,
        save_best_only=True, monitor='val_accuracy', mode='auto')
    csv_logger_3 = tf.keras.callbacks.CSVLogger("model_dsd_op/redense/redense_analysis.csv", separator=',',
                                                append=False)
    model_re_dense.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=sgd_optimizer_3,
                           metrics=['accuracy'])
    model_re_dense.summary()
    history_3 = model_re_dense.fit(
        X_train,
        y_train,
        epochs=epoch_t, validation_data=(X_validation, y_validation),
        callbacks=[checkpointer_5, csv_logger_3, es, checkpointer_6]
    )
    model_re_dense.save_weights(
        "model_dsd_op/redense/redense_model_weights_" + str(epoch_t) + "_" + str(batch_t) + ".hdf5")
    tf.keras.models.save_model(model_re_dense,
                               "model_dsd_op/redense/redense_model_" + str(epoch_t) + "_" + str(batch_t) + "_save")
    with open("model_dsd_op/redense/redense_history.pkl", "wb") as f:
        pickle.dump(history_3.history, f)
    redense_model_accuracy_loss, redense_model_accuracy = model_re_dense.evaluate(
        X_test, y_test, verbose=0)
    print('Baseline test loss:', redense_model_accuracy_loss)
    print('Baseline test accuracy:', redense_model_accuracy)


def get_gzipped_model_size(file):
    _, zipped_file = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(
            zipped_file, "w", compression=zipfile.ZIP_DEFLATED
    ) as f:
        f.write(file)
    return os.path.getsize(zipped_file)


def get_gzipped_model_size_h5(file):
    _, zipped_file = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(
            zipped_file, "w", compression=zipfile.ZIP_DEFLATED
    ) as f:
        f.write(file)
    return os.path.getsize(zipped_file)


def evaluate_tflite_model(filename, x_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    y_pred = []
    for test_image in x_test:
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        y_pred.append(output()[0][0] >= 0.5)
    return (y_pred == np.array(y_test)).mean()

def to_tflite(dataset_path, model_path, path):
    X_test, y_test = load_data_test(dataset_path)
    model_pruned_after = tf.keras.models.load_model(model_path)
    converter_pruned = tf.lite.TFLiteConverter.from_keras_model(model_pruned_after)
    converter_pruned.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]

    model_pruned_tflite = converter_pruned.convert()

    with open(path + ".tflite", "wb") as f:
        f.write(model_pruned_tflite)

    model_pruned_acc = evaluate_tflite_model(
        path + ".tflite", X_test, y_test
    )
    model_pruned_size = get_gzipped_model_size(path + ".tflite")

    print(f"Pruned accuracy: {model_pruned_acc}")
    print(f"Pruned size: {model_pruned_size}")
    # end prunned


def pruned_model_and_train():
    model = tf.keras.models.load_model("model_dsd_op/epoch_100/redense/redense_model_100_32_save")
    X_train, y_train = load_data('Data/unsw_train.csv')
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test, y_test = load_data_test('Data/unsw_test.csv')
    X_validation, y_validation = load_data_validation('Data/unsw_validation.csv')
    X_validation = np.reshape(X_validation, (X_validation.shape[0], 1, X_validation.shape[1]))

    num_images = X_train.shape[0]
    initial_sparsity = 0.25
    final_sparsity = 0.8
    begin_step = 0
    end_step = np.ceil(num_images / 32).astype(np.int32) * 25
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            begin_step=begin_step,
            end_step=end_step)
    }
    model_pruned = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()

    learning_rate = 0.001
    momentum = 0.9
    sgd_optimizer = tf.keras.optimizers.SGD(learning_rate, momentum)
    model_pruned.compile(optimizer=sgd_optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                         metrics=['accuracy'])
    model_pruned.summary()
    prefix = 'dsd_pruned_f_b_32_e_100'
    csv_logger = tf.keras.callbacks.CSVLogger(
        "model_dsd_op/epoch_100/model_pruned/dsd_model_pruned" + prefix + '_trainanalysis.csv', separator=',',
        append=False)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=15, restore_best_weights=True)

    model_pruned.fit(
        X_train,
        y_train,
        epochs=25, validation_data=(X_validation, y_validation),
        callbacks=[csv_logger, es, pruning_callback], verbose=1
    )
    model_pruned_after = tfmot.sparsity.keras.strip_pruning(model_pruned)
    model_pruned_after.compile(optimizer=sgd_optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                               metrics=['accuracy'])
    model_pruned_after.summary()
    pruned_model_accuracy_loss, pruned_model_accuracy = model_pruned.evaluate(
        X_test, y_test, verbose=1)
    strip_pruned_model_accuracy_loss, strip_pruned_model_accuracy = model_pruned_after.evaluate(
        X_test, y_test, verbose=1)
    tf.keras.models.save_model(model_pruned,
                               "model_dsd_op/epoch_100/model_pruned/dsd_model_pruned_model_200_32_save")
    tf.keras.models.save_model(model_pruned_after,
                               "model_dsd_op/epoch_100/model_pruned/dsd_model_pruned_after_model_200_32_save")
    print('Pruned test loss:', pruned_model_accuracy_loss)
    print('Pruned test accuracy:', pruned_model_accuracy)
    print('after strip Pruned test loss:', strip_pruned_model_accuracy_loss)
    print('after strip Pruned test accuracy:', strip_pruned_model_accuracy)


def quantized_redense_model():
    X_test, y_test = load_data_test('Data/unsw_test.csv')
    model_dsd = tf.keras.models.load_model(
        "model_dsd_op/epoch_100/model/redense_model")
    converter = tf.lite.TFLiteConverter.from_keras_model(model_dsd)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    model_quantized_tflite = converter.convert()

    with open("model_dsd_op/epoch_100/model/redense_model_quantized.tflite", "wb") as f:
        f.write(model_quantized_tflite)

    model_quantized_acc = evaluate_tflite_model(
        "model_dsd_op/epoch_100/model/redense_model_quantized.tflite", X_test, y_test
    )
    model_quantized_size = get_gzipped_model_size("model_dsd_op/epoch_100/model/redense_model_quantized.tflite")

    print(f"Quantized accuracy: {model_quantized_acc}")
    print(f"Quantized size: {model_quantized_size}")


def quantized_pruned_model():
    X_test, y_test = load_data_test('Data/unsw_test.csv')
    model_dsd = tf.keras.models.load_model(
        "model_dsd_op/epoch_100/model/prunedmodel")
    converter = tf.lite.TFLiteConverter.from_keras_model(model_dsd)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    model_quantized_tflite = converter.convert()

    with open("model_dsd_op/epoch_100/model/pruned_model_quantized.tflite", "wb") as f:
        f.write(model_quantized_tflite)

    model_quantized_acc = evaluate_tflite_model(
        "model_dsd_op/epoch_100/model/pruned_model_quantized.tflite", X_test, y_test
    )
    model_quantized_size = get_gzipped_model_size("model_dsd_op/epoch_100/model/pruned_model_quantized.tflite")

    print(f"Quantized accuracy: {model_quantized_acc}")
    print(f"Quantized size: {model_quantized_size}")



if __name__ == '__main__':
    train_data()