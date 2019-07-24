import torch
from torch import nn
from torch.utils import model_zoo
from tools.defaultSFA import Model


class BaseConv(nn.Module):

    def __init__(self, inplanes, outplanes,k_size,padding):
        super(BaseConv, self).__init__()

        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=k_size, padding=padding,stride=1)
        self.use_bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.use_bn(x)
        x = self.relu(x)
        return x


class PruningNet(nn.Module):
    def __init__(self):
        super(PruningNet, self).__init__()

        # input = self.inputs
        # h = self.h
        # w = self.w
        # self.height = 512
        # self.height = 512

        self.conv0_0 = BaseConv(3,32,3,1)
        self.conv0_1 = BaseConv(32,32,3,1)
        self.skip0 = BaseConv(32,8,1,0)

        self.maxpool0 = nn.MaxPool2d(2, 2)
        self.conv1_0 = BaseConv(32,64,3,1)
        self.conv1_1 = BaseConv(64,64,3,1)
        self.skip1 = BaseConv(64,8,1,0)

        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2_0 = BaseConv(64, 128, 3,1)
        self.conv2_1 = BaseConv(128, 128, 3,1)
        self.skip2 = BaseConv(128, 16, 1,0)

        self.skipup = nn.UpsamplingBilinear2d([512,512])

        self.upconv0 = BaseConv(32,32,3,1)
        self.output = nn.Conv2d(32, 1, kernel_size=3, padding=1,stride=1)


    def forward(self, input):
        x = self.conv0_0(input)
        x = self.conv0_1(x)
        skip0 = self.skip0(x)

        x = self.maxpool0(x)
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        skip1 = self.skip1(x)

        x = self.maxpool1(x)
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        skip2 = self.skip2(x)

        skip2up = self.skipup(skip2)
        skip1up = self.skipup(skip1)

        upcat = torch.cat([skip0, skip1up,skip2up], 1)
        upconv0 = self.upconv0(upcat)
        output = self.output(upconv0)

        return output

if __name__ == '__main__':
    input = torch.rand(1,3,512,512)
    model = PruningNet()
    output = model(input)

    print(output.size())



        # input = self.vgg(input)
        #
        # amp_out = self.amp(*input)
        # dmp_out = self.dmp(*input)
        #
        # amp_out = self.conv_att(amp_out)
        # dmp_out = amp_out * dmp_out
        # dmp_out = self.conv_out(dmp_out)

        # return dmp_out, amp_out

# def forward4_1(self, imgmean):
#     with tf.variable_scope("TDGNet"):
#         input = self.inputs
#         h = self.h
#         w = self.w
#         imgmean = tf.reshape(imgmean, [-1, 1, 1, 3])
#         # newmean = 100*tf.get_variable('newmeanu, shape=[1,1,1,3], trainable=self.trainable)
#         # newmean = 100 * tf.Variable([[[[1.2, 0.8, 0.6]]]], dtype=tf.float32, trainable=self.trainable, name='newmean')
#         # newmean = tf.constant([[[[138., 100., 70.]]]], dtype=tf.float32)
#         newmean = tf.constant([[[[145., 105, 75.]]]], dtype=tf.float32)
#         input = input * newmean / imgmean
#
#         x = self.conv(input, 3, 3, 32, 1, 1, relu=True, name='conv0_0', padding='SAME')
#         x = self.conv(x,     3, 3, 32, 1, 1, relu=True, name='conv0_1', padding='SAME')
#         skip0 = self.conv(x, 1, 1, 8, 1, 1, relu=True, name='skip0')
#
#         x = self.max_pool(x, 2, 2, 2, 2, name='maxpool0', padding='SAME')
#         x = self.conv(x, 3, 3, 64, 1, 1, relu=True, name='conv1_0', padding='SAME')
#         x = self.conv(x, 3, 3, 64, 1, 1, relu=True, name='conv1_1', padding='SAME')
#         skip1 = self.conv(x, 1, 1, 8, 1, 1, relu=True, name='skip1')
#
#         x = self.max_pool(x, 2, 2, 2, 2, name='maxpool1', padding='SAME')
#         x = self.conv(x, 3, 3, 128, 1, 1, relu=True, name='conv2_0', padding='SAME')
#         x = self.conv(x, 3, 3, 128, 1, 1, relu=True, name='conv2_1', padding='SAME')
#         skip2 = self.conv(x, 3, 3, 16, 1, 1, relu=True, name='skip2')
#
#         skip2up = tf.image.resize_bilinear(skip2, size=[h, w], align_corners=True, name='skip2up')
#         skip1up = tf.image.resize_bilinear(skip1, size=[h, w], align_corners=True, name='skip1up')
#         upcat = tf.concat([skip0, skip1up, skip2up], axis=3, name='upcat')
#         upconv0 = self.conv(upcat, 3, 3, 32, 1, 1, relu=True, name='conv_up0')
#         # upconv1 = self.conv(upconv0, 3, 3, 32, 1, 1, relu=True, name='conv_up1')
#         output = self.conv(upconv0, 3, 3, 1, 1, 1, relu=False, name='output')
#     return output
