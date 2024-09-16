import os
import tf_keras as keras
import numpy as np
import tensorflow as tf

MAX_LEN = 174
os.environ["TF_USE_LEGACY_KERAS"]="1"

embedding_matrix_one_hot = np.array([[0, 0, 0, 0],
                                     [1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])


def binary_focal_loss(gamma=2., alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (keras.backend.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (keras.backend.ones_like(y_true) - y_true) * (
                    keras.backend.ones_like(y_true) - y_pred) + keras.backend.epsilon()
        focal_loss = - alpha_t * keras.backend.pow((keras.backend.ones_like(y_true) - p_t), gamma) * keras.backend.log(p_t)
        return keras.backend.mean(focal_loss)

    return binary_focal_loss_fixed


def ResBlock1(x, filters, kernel_size1, kernel_size2, dilation_rate):
    # r = tfa.layers.WeightNormalization(Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu'))(x)#第一卷积
    r1 = keras.layers.Conv1D(filters, kernel_size1, padding='same', dilation_rate=dilation_rate)(x)
    r2 = keras.layers.Conv1D(filters, kernel_size2, padding='same', dilation_rate=dilation_rate)(x)
    r1 = keras.layers.BatchNormalization()(r1)
    # r = SpatialDropout1D(0.2)(r)
    r1 = keras.layers.Dropout(0.2)(r1)
    r1 = keras.layers.Activation('relu')(r1)
    r2 = keras.layers.BatchNormalization()(r2)
    # r = SpatialDropout1D(0.2)(r)
    r2 = keras.layers.Dropout(0.2)(r2)
    r2 = keras.layers.Activation('relu')(r2)
    r = keras.layers.concatenate([r1, r2])
    r3 = keras.layers.Conv1D(filters, kernel_size1, padding='same', dilation_rate=dilation_rate)(r)
    r4 = keras.layers.Conv1D(filters, kernel_size2, padding='same', dilation_rate=dilation_rate)(r)
    r3 = keras.layers.BatchNormalization()(r3)
    # r = SpatialDropout1D(0.2)(r)
    r3 = keras.layers.Dropout(0.2)(r3)
    r3 = keras.layers.Activation('relu')(r3)
    r4 = keras.layers.BatchNormalization()(r4)
    r4 = keras.layers.Dropout(0.2)(r4)
    r4 = keras.layers.Activation('relu')(r4)
    r = keras.layers.concatenate([r3, r4], name='concat%d' % dilation_rate)
    # r = tfa.layers.WeightNormalization(Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu'))(r)#第二卷积
    # r = SpatialDropout1D(0.2)(r)
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = keras.layers.Conv1D(filters * 2, 1, padding='same')(x)  # shortcut（捷径）
    o = keras.layers.add([r, shortcut])
    o = keras.layers.Activation('relu')(o)  # 激活函数
    return o


def ResBlock(x, filters, kernel_size, dilation_rate):
    # r = tfa.layers.WeightNormalization(Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu'))(x)#第一卷积
    r = keras.layers.Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    r = keras.layers.BatchNormalization()(r)
    # r = SpatialDropout1D(0.2)(r)
    r = keras.layers.Dropout(0.2)(r)
    r = keras.layers.Activation('relu')(r)
    r = keras.layers.Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)
    r = keras.layers.BatchNormalization()(r)
    # r = SpatialDropout1D(0.2)(r)
    r = keras.layers.Dropout(0.2)(r)
    r = keras.layers.Activation('relu')(r)
    # r = tfa.layers.WeightNormalization(Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu'))(r)#第二卷积
    # r = SpatialDropout1D(0.2)(r)
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = keras.layers.Conv1D(filters, 1, padding='same')(x)  # shortcut（捷径）
    o = keras.layers.add([r, shortcut])
    o = keras.layers.Activation('relu')(o)  # 激活函数
    return o


class AttLayer(keras.layers.Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.trainable_weight = None
        self.init = tf.compat.v2.random_normal_initializer(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = keras.backend.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = keras.backend.variable(self.init((self.attention_dim,)))
        self.u = keras.backend.variable(self.init((self.attention_dim, 1)))
        self.trainable_weight = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = keras.backend.tanh(tf.nn.bias_add(keras.backend.dot(x, self.W), self.b))
        ait = keras.backend.dot(uit, self.u)
        ait = keras.backend.squeeze(ait, -1)

        ait = keras.backend.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= keras.backend.cast(mask, keras.backend.floatx())
        ait /= keras.backend.cast(keras.backend.sum(ait, axis=1, keepdims=True) +
                              keras.backend.epsilon(), keras.backend.floatx())
        ait = keras.backend.expand_dims(ait)
        weighted_input = x * ait
        return keras.backend.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def CNN_GRU_ATT_model(layers=2, filters=16,
                      growth_rate=8, dropout_rate=0.2, weight_decay=1e-4):
    sequence = keras.layers.Input(shape=(MAX_LEN,))

    emb_en = keras.layers.Embedding(5, 4, weights=[embedding_matrix_one_hot],
                                    trainable=False)(sequence)
    conv_layer1 = keras.layers.Convolution1D(
        filters=64,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer1 = keras.layers.MaxPooling1D(pool_size=2)
    conv_layer2 = keras.layers.Convolution1D(
        filters=16,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer2 = keras.layers.MaxPooling1D(pool_size=2)
    conv_layer3 = keras.layers.Convolution1D(
        filters=16,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer3 = keras.layers.MaxPooling1D(pool_size=2)

    enhancer_branch = keras.Sequential()
    enhancer_branch.add(conv_layer1)
    enhancer_branch.add(keras.layers.Activation("relu"))
    enhancer_branch.add(max_pool_layer1)
    enhancer_branch.add(keras.layers.BatchNormalization())
    enhancer_branch.add(keras.layers.Dropout(0.2))
    enhancer_branch.add(conv_layer2)
    enhancer_branch.add(keras.layers.Activation("relu"))
    enhancer_branch.add(max_pool_layer2)
    enhancer_branch.add(keras.layers.BatchNormalization())
    enhancer_branch.add(keras.layers.Dropout(0.2))
    # enhancer_branch.add(conv_layer3)
    # enhancer_branch.add(keras.layers.Activation("relu"))
    # enhancer_branch.add(max_pool_layer3)
    # enhancer_branch.add(keras.layers.BatchNormalization())
    # enhancer_branch.add(keras.layers.Dropout(0.2))
    enhancer_out = enhancer_branch(emb_en)
    l_gru1 = keras.layers.Bidirectional(keras.layers.GRU(16, return_sequences=True))(enhancer_out)
    x = AttLayer(16)(l_gru1)

    dt = keras.layers.Dense(32)(x)  # kernel_initializer="glorot_uniform"
    dt = keras.layers.BatchNormalization()(dt)
    dt = keras.layers.Activation("relu")(dt)
    dt = keras.layers.Dropout(0.2)(dt)
    preds = keras.layers.Dense(1, activation='sigmoid')(dt)
    model = keras.Model([sequence], preds)
    return model


def deepires_model(layers=2, filters=16,
                   growth_rate=8, dropout_rate=0.2, weight_decay=1e-4):
    sequence = keras.layers.Input(shape=(MAX_LEN,))
    emb_en = keras.layers.Embedding(5, 4, weights=[embedding_matrix_one_hot],
                                    trainable=False)(sequence)
    x = ResBlock1(emb_en, filters=16, kernel_size1=2, kernel_size2=3, dilation_rate=1)
    x = ResBlock1(x, filters=8, kernel_size1=2, kernel_size2=3, dilation_rate=2)
    l_gru1 = keras.layers.Bidirectional(keras.layers.GRU(8, return_sequences=True))(x)
    x = AttLayer(8)(l_gru1)

    dt = keras.layers.Dense(32)(x)  # kernel_initializer="glorot_uniform"
    dt = keras.layers.BatchNormalization()(dt)
    dt = keras.layers.Activation("relu")(dt)
    dt = keras.layers.Dropout(0.2)(dt)
    preds = keras.layers.Dense(1, activation='sigmoid')(dt)
    return keras.Model([sequence], preds)


def CNN_GRU_model(layers=2, filters=16,
                  growth_rate=8, dropout_rate=0.2, weight_decay=1e-4):
    sequence = keras.layers.Input(shape=(MAX_LEN,))

    emb_en = keras.layers.Embedding(5, 4, weights=[embedding_matrix_one_hot],
                                    trainable=False)(sequence)
    conv_layer1 = keras.layers.Convolution1D(
        filters=32,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer1 = keras.layers.MaxPooling1D(pool_size=int(2))
    conv_layer2 = keras.layers.Convolution1D(
        filters=16,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer2 = keras.layers.MaxPooling1D(pool_size=int(2))
    conv_layer3 = keras.layers.Convolution1D(
        filters=16,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer3 = keras.layers.MaxPooling1D(pool_size=int(2))

    enhancer_branch = keras.Sequential()
    enhancer_branch.add(conv_layer1)
    enhancer_branch.add(keras.layers.Activation("relu"))
    enhancer_branch.add(max_pool_layer1)
    enhancer_branch.add(keras.layers.BatchNormalization())
    enhancer_branch.add(keras.layers.Dropout(0.2))
    enhancer_branch.add(conv_layer2)
    enhancer_branch.add(keras.layers.Activation("relu"))
    enhancer_branch.add(max_pool_layer2)
    enhancer_branch.add(keras.layers.BatchNormalization())
    enhancer_branch.add(keras.layers.Dropout(0.2))
    # enhancer_branch.add(conv_layer3)
    # enhancer_branch.add(keras.layers.Activation("relu"))
    # enhancer_branch.add(max_pool_layer3)
    # enhancer_branch.add(keras.layers.BatchNormalization())
    # enhancer_branch.add(keras.layers.Dropout(0.2))
    enhancer_out = enhancer_branch(emb_en)
    l_gru1 = keras.layers.Bidirectional(keras.layers.GRU(16, return_sequences=True))(enhancer_out)
    x = keras.layers.Flatten()(l_gru1)

    dt = keras.layers.Dense(32)(x)  # kernel_initializer="glorot_uniform"
    dt = keras.layers.BatchNormalization()(dt)
    dt = keras.layers.Activation("relu")(dt)
    dt = keras.layers.Dropout(0.2)(dt)
    preds = keras.layers.Dense(1, activation='sigmoid')(dt)
    return keras.Model([sequence], preds)


def CNN_model(layers=2, filters=16,
              growth_rate=8, dropout_rate=0.2, weight_decay=1e-4):
    sequence = keras.layers.Input(shape=(MAX_LEN,))

    emb_en = keras.layers.Embedding(5, 4, weights=[embedding_matrix_one_hot],
                                    trainable=False)(sequence)
    conv_layer1 = keras.layers.Convolution1D(
        filters=64,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer1 = keras.layers.MaxPooling1D(pool_size=int(2))
    conv_layer2 = keras.layers.Convolution1D(
        filters=32,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer2 = keras.layers.MaxPooling1D(pool_size=int(2))
    conv_layer3 = keras.layers.Convolution1D(
        filters=16,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer3 = keras.layers.MaxPooling1D(pool_size=int(2))

    enhancer_branch = keras.Sequential()
    enhancer_branch.add(conv_layer1)
    enhancer_branch.add(keras.layers.Activation("relu"))
    enhancer_branch.add(max_pool_layer1)
    enhancer_branch.add(keras.layers.BatchNormalization())
    enhancer_branch.add(keras.layers.Dropout(0.2))
    enhancer_branch.add(conv_layer2)
    enhancer_branch.add(keras.layers.Activation("relu"))
    enhancer_branch.add(max_pool_layer2)
    enhancer_branch.add(keras.layers.BatchNormalization())
    enhancer_branch.add(keras.layers.Dropout(0.2))
    # enhancer_branch.add(conv_layer3)
    # enhancer_branch.add(keras.layers.Activation("relu"))
    # enhancer_branch.add(max_pool_layer3)
    # enhancer_branch.add(keras.layers.BatchNormalization())
    # enhancer_branch.add(keras.layers.Dropout(0.2))
    enhancer_out = enhancer_branch(emb_en)
    flatten = keras.layers.Flatten()(enhancer_out)
    dt1 = keras.layers.Dropout(0.2)(flatten)

    dt = keras.layers.Dense(64)(dt1)  # kernel_initializer="glorot_uniform"
    dt = keras.layers.BatchNormalization()(dt)
    dt = keras.layers.Activation("relu")(dt)
    dt = keras.layers.Dropout(0.2)(dt)
    preds = keras.layers.Dense(1, activation='sigmoid')(dt)
    return keras.Model([sequence], preds)


def GRU_model(layers=2, filters=16,
              growth_rate=8, dropout_rate=0.2, weight_decay=1e-4):
    sequence = keras.layers.Input(shape=(MAX_LEN,))
    # enhancers1 = keras.layers.Input(shape=(MAX_LEN_en,7))

    emb_en = keras.layers.Embedding(5, 4, weights=[embedding_matrix_one_hot],
                                    trainable=False)(sequence)

    # x_multiply2 = SimAM()(enhancer_out)
    l_gru1 = keras.layers.Bidirectional(keras.layers.GRU(16, return_sequences=True))(emb_en)
    # l_gru2 = keras.layers.Bidirectional(GRU(8, return_sequences=True))(l_gru1)
    x = AttLayer(16)(l_gru1)

    bn2 = keras.layers.BatchNormalization()(x)

    dt1 = keras.layers.Dropout(0.2)(bn2)

    dt = keras.layers.Dense(64)(dt1)  # kernel_initializer="glorot_uniform"
    dt = keras.layers.BatchNormalization()(dt)
    dt = keras.layers.Activation("relu")(dt)
    dt = keras.layers.Dropout(0.2)(dt)
    preds = keras.layers.Dense(1, activation='sigmoid')(dt)
    return keras.Model([sequence], preds)
