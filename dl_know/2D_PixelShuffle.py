import torch.nn as nn
import torch


class PixelShuffle(nn.Module):
    """
    2维PixelShuffle模块
    """
    def __init__(self, upscale_factor):
        """
        :param upscale_factor: tensor的放大倍数
        """
        super(PixelShuffle, self).__init__()

        self.upscale_factor = upscale_factor

    def forward(self, inputs):

        batch_size, channels, in_height, in_width = inputs.size()

        channels //= self.upscale_factor ** 2

        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels,self.upscale_factor,self.upscale_factor,in_height, in_width)

        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()

        return shuffle_out.view(batch_size, channels, out_height, out_width)


# def pixel_shuffle(upsacle_factor):
#     return PixelShuffle(upsacle_factor).cuda()


import torch
from time import time

upscale_factor = 2

# cpu
# ps = PixelShuffle(upscale_factor)
# inputData = torch.rand(1,512*upscale_factor**2,10,9)

# gpu
ps = PixelShuffle(upscale_factor).cuda()
inputData = torch.rand(1,512*upscale_factor**2,10,9).cuda()

# 测试模块效率(运行时间)
start = time()
output = ps(inputData)

print(time() - start)
print(inputData.size(), output.size())

# cpu运行时间
# 0.0011870861053466797

# gpu运行时间
# 0.00031495094299316406


