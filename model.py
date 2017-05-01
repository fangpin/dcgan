import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import InputLayer, Conv2d, BatchNormLayer, \
    FlattenLayer, DenseLayer, ReshapeLayer, DeConv2d

class Cfg(object):

    def __init__(self):
        self.channels = 3
        self.batch_size = 64
        self.img_size = 64
        self.z_dim = 100
        self.sample_size = 64
        self.lr = 0.0002
        self.beta1 = 0.5
        self.checkpoint_dir = 'checkpoint'
        self.sample_dir = 'samples'
        self.dataset = 'CelebA'
        self.epoch = 25
        self.crop_size = 108
        self.is_crop = True
        self.sample_step = 500
        self.save_step = 500


CFG = Cfg()


def generator(inputs, is_train=True, reuse=False):
    img_size = CFG.img_size
    s2, s4, s8, s16 = [int(img_size/i) for i in [2, 4, 8, 16]]
    gfs = 64
    channels = CFG.channels
    batch_size = CFG.batch_size

    W_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope('generator', reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        g = InputLayer(inputs, name='g/inputs')
        g = DenseLayer(g, gfs*8*s16*s16, W_init=W_init, act=tl.act.identity, name='g/fc1')
        g = ReshapeLayer(g, shape=(-1, s16, s16, gfs*8), name='g/reshape2')
        g = BatchNormLayer(g, act=tf.nn.relu, is_train=is_train,
                           gamma_init=gamma_init, name='g/bn3')

        g = DeConv2d(g, gfs*4, (5, 5), out_size=(s8, s8), strides=(2, 2),
                     batch_size=batch_size, act=None, W_init=W_init,
                     name='g/dconv4')
        g = BatchNormLayer(g, act=tf.nn.relu, is_train=is_train,
                           gamma_init=gamma_init, name='g/bn5')

        g = DeConv2d(g, gfs*2, (5, 5), out_size=(s4, s4), strides=(2, 2),
                     batch_size=batch_size, act=None, W_init=W_init,
                     name='g/dconv6')
        g = BatchNormLayer(g, act=tf.nn.relu, is_train=is_train,
                           gamma_init=gamma_init, name='g/bn7')

        g = DeConv2d(g, gfs, (5, 5), out_size=(s2, s2),
                     strides=(2, 2), batch_size=batch_size, act=None,
                     W_init=W_init, name='g/dconv8')
        g = BatchNormLayer(g, act=tf.nn.relu, is_train=is_train,
                           gamma_init=gamma_init, name='g/bn9')

        g = DeConv2d(g, channels, (5, 5), out_size=(img_size, img_size),
                     strides=(2, 2), batch_size=batch_size, act=None,
                     W_init=W_init, name='g/dconv10')

        logits = g.outputs
        g.outputs = tf.nn.tanh(g.outputs)
    return g, logits


def discriminator(inputs, is_train=True, reuse=False):
    dfs = 64
    gamma_init = tf.random_normal_initializer(1., 0.02)
    W_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope('discriminator', reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        d = InputLayer(inputs, name='d/inputs')
        d = Conv2d(d, dfs, (5, 5), (2, 2), W_init=W_init,
                   act=lambda x: tl.act.lrelu(x, 0.2), name='d/conv1')

        d = Conv2d(d, dfs*2, (5, 5), (2, 2), W_init=W_init,
                   act=None, name='d/conv2')
        d = BatchNormLayer(d, act=lambda x: tl.act.lrelu(x, 0.2),
                           is_train=is_train, gamma_init=gamma_init, name='d/bn3')

        d = Conv2d(d, dfs*4, (5, 5), (2, 2), W_init=W_init,
                   act=None, name='d/conv4')
        d = BatchNormLayer(d, act=lambda x: tl.act.lrelu(x, 0.2),
                           is_train=is_train, gamma_init=gamma_init, name='d/bn5')

        d = Conv2d(d, dfs*8, (5, 5), (2, 2), W_init=W_init,
                   act=None, name='d/conv6')
        d = BatchNormLayer(d, act=lambda x: tl.act.lrelu(x, 0.2),
                           is_train=is_train, gamma_init=gamma_init, name='d/bn7')

        d = FlattenLayer(d, name='d/flt8')
        d = DenseLayer(d, 1, act=tl.act.identity, W_init=W_init, name='d/output')

        logits = d.outputs
        d.outputs = tf.nn.sigmoid(d.outputs)
        return d, logits
