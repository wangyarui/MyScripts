import tensorflow as tf
import numpy as np

# slim = tf.contrib.slim

# https://blog.csdn.net/qq_37482202/article/details/84978224
def conv_op(input_op, name, kh, kw, n_out, dh, dw):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)

def pruneVGG(input_op):

    conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3, n_out=12, dh=1, dw=1)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3, n_out=17, dh=1, dw=1)
    pooll = mpool_op(conv1_2, name='pool1', kh=2, kw=2, dw=2, dh=2)

    conv2_1 = conv_op(pooll, name='conv2_1', kh=3, kw=3, n_out=39, dh=1, dw=1)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=54, dh=1, dw=1)
    pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dh=2, dw=2)

    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=49, dh=1, dw=1)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=26, dh=1, dw=1)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=71, dh=1, dw=1)
    pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dh=2, dw=2)

    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=46, dh=1, dw=1)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=42, dh=1, dw=1)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=39, dh=1, dw=1)
    pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dh=2, dw=2)

    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=154, dh=1, dw=1)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=123, dh=1, dw=1)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=510, dh=1, dw=1)

    return conv2_2,conv3_3, conv4_3, conv5_3


def BackEnd(input):
    conv2_2, conv3_3, conv4_3, conv5_3 = input

    input = tf.image.resize_bilinear(conv5_3,size=[conv5_3.shape[1] * 2,conv5_3.shape[2] * 2],
                                     align_corners=True, name='upsample0_0')

    input = tf.concat([input, conv4_3],3)
    input = conv_op(input, name='BEconv1_0', kh=1, kw=1, n_out=256, dh=1, dw=1)
    input = conv_op(input, name='BEconv2_0', kh=3, kw=3, n_out=256, dh=1, dw=1)
    input = tf.image.resize_bilinear(input, size=[input.shape[1] * 2, input.shape[2] * 2],
                                     align_corners=True,name='upsample1_0')

    input = tf.concat([input, conv3_3], 3)
    input = conv_op(input, name='BEconv3_0', kh=1, kw=1, n_out=128, dh=1, dw=1)
    input = conv_op(input, name='BEconv4_0', kh=3, kw=3, n_out=128, dh=1, dw=1)
    input = tf.image.resize_bilinear(input, size=[input.shape[1] * 2, input.shape[2] * 2],
                                     align_corners=True, name='upsample1_0')

    input = tf.concat([input, conv2_2], 3)
    input = conv_op(input, name='BEconv5_0', kh=1, kw=1, n_out=64, dh=1, dw=1)
    input = conv_op(input, name='BEconv6_0', kh=3, kw=3, n_out=64, dh=1, dw=1)
    input = conv_op(input, name='BEconv7_0', kh=3, kw=3, n_out=32, dh=1, dw=1)

    return input

def pruneSFANet(input):
    input = pruneVGG(input)

    dmp_out = BackEnd(input)
    amp_out = BackEnd(input)

    amp_out = conv_op(amp_out,name='amp_out', kh=1, kw=1, n_out=1, dh=1, dw=1)
    dmp_out = amp_out * dmp_out
    dmp_out = conv_op(dmp_out,name='dmp_out', kh=1, kw=1, n_out=1, dh=1, dw=1)

    return dmp_out,amp_out


if __name__ == '__main__':
    from PIL import Image
    import time
    from tensorflow.python.framework import graph_util
    import matplotlib.pylab as plt

    image = Image.open('/home/osk/图片/IMG_4.jpg')
    image = image.resize([400,400])
    image_array = np.array(image)

    image_array = image_array.astype(float)

    image = np.reshape(image_array,[1,400,400,3])

    x = tf.placeholder(tf.float32,[400, 400,3])
    x = tf.reshape(x, [1,400,400,3])
    net = pruneSFANet(x)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        # image_resize.eval()
        # img_run = sess.run(image_resize)
        time1 = time.time()

        density_map,attention_map= sess.run(net, feed_dict={x: image})
        print(time.time()-time1)
        density_map = np.asarray(density_map)
        attention_map= np.asarray(attention_map)

        saver.save(sess, "model/pruneSFANet")

        print(density_map.shape)
        print(attention_map.shape)

        # image_ndarray = density_map.eval(session=sess)
        test=np.squeeze(density_map,0)
        test=np.squeeze(test,-1)
        print(type(test))
        print(test.shape)
        plt.imshow(test)
        plt.show()











# # Testing the data flow of the network with some random inputs.
# if __name__ == "__main__":
#     x = tf.placeholder(tf.float32, [1, 200, 300, 1])
#     net = build(x)
#     init = tf.initialize_all_variables()
#     sess = tf.Session()
#     sess.run(init)
#     d_map = sess.run(net, feed_dict={x: 255 * np.ones(shape=(1, 200, 300, 1), dtype=np.float32)})
#     prediction = np.asarray(d_map)
#     prediction = np.squeeze(prediction, axis=0)
#     prediction = np.squeeze(prediction, axis=2)


# class PruningModel(nn.Module):
#     def __init__(self,cfg=None):
#         super(PruningModel, self).__init__()
#         self.vgg = VGG(cfg=cfg)
#         # self.load_vgg()
#
#         self.amp = BackEnd()
#         self.dmp = BackEnd()
#
#         self.conv_att = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=True)
#         self.conv_out = BaseConv(32, 1, 1, 1, activation=None, use_bn=False)
#
#     def forward(self, input):
#         input = self.vgg(input)
#
#         amp_out = self.amp(*input)
#         dmp_out = self.dmp(*input)
#
#         amp_out = self.conv_att(amp_out)
#         dmp_out = amp_out * dmp_out
#         dmp_out = self.conv_out(dmp_out)
#
#         return dmp_out, amp_out
#
#
# class VGG(nn.Module):
#     def __init__(self,cfg=None):
#         super(VGG, self).__init__()
#
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv1_1 = BaseConv(3, 12, 3, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv1_2 = BaseConv(12, 17, 3, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv2_1 = BaseConv(17, 39, 3, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv2_2 = BaseConv(39, 54, 3, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv3_1 = BaseConv(54, 49, 3, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv3_2 = BaseConv(49, 26, 3, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv3_3 = BaseConv(26, 71, 3, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv4_1 = BaseConv(71, 46, 3, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv4_2 = BaseConv(46, 42, 3, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv4_3 = BaseConv(42, 39, 3, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv5_1 = BaseConv(39, 154, 3, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv5_2 = BaseConv(154, 123, 3, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv5_3 = BaseConv(123, 510, 3, 1, activation=nn.ReLU(), use_bn=True)
#
#
#     def forward(self, input):
#         input = self.conv1_1(input)
#         input = self.conv1_2(input)
#         input = self.pool(input)
#         input = self.conv2_1(input)
#         conv2_2 = self.conv2_2(input)
#
#         input = self.pool(conv2_2)
#         input = self.conv3_1(input)
#         input = self.conv3_2(input)
#         conv3_3 = self.conv3_3(input)
#
#         input = self.pool(conv3_3)
#         input = self.conv4_1(input)
#         input = self.conv4_2(input)
#         conv4_3 = self.conv4_3(input)
#
#         input = self.pool(conv4_3)
#         input = self.conv5_1(input)
#         input = self.conv5_2(input)
#         conv5_3 = self.conv5_3(input)
#
#         return conv2_2, conv3_3, conv4_3, conv5_3
#
#
# class BackEnd(nn.Module):
#     def __init__(self):
#         super(BackEnd, self).__init__()
#         self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
#
#         self.conv1 = BaseConv(549, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
#
#         self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
#
#         self.conv3 = BaseConv(327, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
#
#         self.conv5 = BaseConv(182, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
#         self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
#
#     def forward(self, *input):
#         conv2_2, conv3_3, conv4_3, conv5_3 = input
#
#         input = self.upsample(conv5_3)
#
#         input = torch.cat([input, conv4_3], 1)
#         input = self.conv1(input)
#         input = self.conv2(input)
#         input = self.upsample(input)
#
#         input = torch.cat([input, conv3_3], 1)
#         input = self.conv3(input)
#         input = self.conv4(input)
#         input = self.upsample(input)
#
#         input = torch.cat([input, conv2_2], 1)
#         input = self.conv5(input)
#         input = self.conv6(input)
#         input = self.conv7(input)
#
#         return input
#
#
# class BaseConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
#         super(BaseConv, self).__init__()
#         self.use_bn = use_bn
#         self.activation = activation
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
#         self.conv.weight.data.normal_(0, 0.01)
#         self.conv.bias.data.zero_()
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.bn.weight.data.fill_(1)
#         self.bn.bias.data.zero_()
#
#     def forward(self, input):
#         input = self.conv(input)
#         if self.use_bn:
#             input = self.bn(input)
#         if self.activation:
#             input = self.activation(input)
#
#         return input