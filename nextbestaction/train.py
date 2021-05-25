import pickle
import nextbestaction.util as util
from datetime import datetime
from sklearn.neighbors import BallTree
import tensorflow as tf


def train(args, preprocess_manager):

    batch_size = args.batch_size_train

    util.ll_print("Loading Data starts... \n")
    X, Y_Event, Y_Time, sequence_max_length, num_features_all, num_features_activities = preprocess_manager.create_and_encode_training_set(args)

    util.ll_print('\n Build model for suffix prediction... \n')
    if args.dnn_architecture == 0:
        # train a two-layer LSTM with one shared layer
        main_input = tf.keras.layers.Input(shape=(sequence_max_length, num_features_all), name='main_input')
        # the shared layer
        l1 = tf.keras.layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input)
        b1 = tf.keras.layers.BatchNormalization()(l1)
        # the layer specialized in activity prediction
        l2_1 = tf.keras.layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1)
        b2_1 = tf.keras.layers.BatchNormalization()(l2_1)
        # the layer specialized in time prediction
        l2_2 = tf.keras.layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1)
        b2_2 = tf.keras.layers.BatchNormalization()(l2_2)

    event_output = tf.keras.layers.Dense(num_features_activities + 1, activation='softmax', kernel_initializer='glorot_uniform', name='event_output')(b2_1)
    time_output = tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)
    model_suffix_prediction = tf.keras.models.Model(inputs=[main_input], outputs=[event_output, time_output])

    opt = tf.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004, clipvalue=3)

    model_suffix_prediction.compile(loss={'event_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=opt)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%smodel_suffix_prediction_%s.h5' % (args.checkpoint_dir, preprocess_manager.iteration_cross_validation), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    model_suffix_prediction.summary()

    start_training_time = datetime.now()
    model_suffix_prediction.fit(X, {'event_output': Y_Event, 'time_output': Y_Time},
                                validation_split=1 / args.num_folds, verbose=1,
                                callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=batch_size,
                                epochs=args.dnn_num_epochs)
    training_time = datetime.now() - start_training_time

    if args.next_best_action:

        util.ll_print('Build model for candidate determination... \n')
        X_case_based_suffix = preprocess_manager.transform_tensor_to_matrix(X)
        model_candidate_selection = BallTree(X_case_based_suffix, leaf_size=2)  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
        pickle.dump(model_candidate_selection, open("%smodel_candidate_selection_%s" % (args.checkpoint_dir, preprocess_manager.iteration_cross_validation), 'wb'))

    return training_time.total_seconds()
