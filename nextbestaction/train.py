import pickle
from datetime import datetime
from sklearn.neighbors import BallTree
import tensorflow as tf


def train(args, preprocess_manager):

    print("Loading Data starts... \n")
    X, Y_act, Y_time, seq_max_length, num_features_all, num_features_acts = preprocess_manager.create_and_encode_training_set()

    print('\n Build model for suffix prediction... \n')
    main_input = tf.keras.layers.Input(shape=(seq_max_length, num_features_all), name='main_input')
    l1 = tf.keras.layers.LSTM(100, return_sequences=True, dropout=0.2)(main_input)
    b1 = tf.keras.layers.BatchNormalization()(l1)
    l2_1 = tf.keras.layers.LSTM(100, return_sequences=False, dropout=0.2)(b1)
    b2_1 = tf.keras.layers.BatchNormalization()(l2_1)
    l2_2 = tf.keras.layers.LSTM(100, return_sequences=False, dropout=0.2)(b1)
    b2_2 = tf.keras.layers.BatchNormalization()(l2_2)

    act_output = tf.keras.layers.Dense(num_features_acts + 1, activation='softmax', name='act_output')(b2_1)
    time_output = tf.keras.layers.Dense(1, name='time_output')(b2_2)
    model_suffix_prediction = tf.keras.models.Model(inputs=[main_input], outputs=[act_output, time_output])
    opt = tf.optimizers.Nadam(lr=args.learning_rate, epsilon=1e-8, schedule_decay=0.004, clipvalue=3)
    model_suffix_prediction.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=opt)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%smodel_suffix_prediction.h5' % args.checkpoint_dir,
                                                          monitor='val_loss', save_best_only=True)
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5)
    model_suffix_prediction.summary()

    start_training_time = datetime.now()
    model_suffix_prediction.fit(X, {'act_output': Y_act, 'time_output': Y_time},
                                validation_split=args.validation_size,
                                callbacks=[early_stopping, model_checkpoint, lr_reducer],
                                batch_size=args.batch_size_train,
                                epochs=args.dnn_num_epochs)
    training_time = datetime.now() - start_training_time

    if args.next_best_action:
        print('Build model for candidate determination... \n')
        X_case_based_suffix = preprocess_manager.transform_tensor_to_matrix(X)
        model_candidate_selection = BallTree(X_case_based_suffix, leaf_size=2)  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
        pickle.dump(model_candidate_selection, open("%smodel_candidate_selection" % args.checkpoint_dir, 'wb'))

    return training_time.total_seconds()
