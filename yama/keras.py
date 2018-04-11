import keras.backend as K


def limit_memory():
    K.get_session().close()
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=config))
