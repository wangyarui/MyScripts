# model结构
import torch 
#-------------------------------------------------------------------------
class FastPose_SE(nn.Module):
    conv_dim = 128

    def __init__(self):
        super(FastPose_SE, self).__init__()

        self.preact = SERes2net('resnet50')
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)

        self.conv_out = nn.Conv2d(
            self.conv_dim, opt.nClasses, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.preact(x)
        # out = self.conv0(out)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        return out
#-------------------------------------------------------------------------
loadModel = "model path"
m = FastPose_SE().cuda()

# 冻结权重
pretrained_dict = torch.load(loadModel)
model_dict = m.state_dict()
pretrained_dict = {k: v for k,v in pretrained_dict.items() if 'conv_out' in model_dict}
model_dict.update(pretrained_dict)
m.load_state_dict(model_dict)
