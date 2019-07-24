import torch
from torch.utils import data
from dataset import Dataset
# from tools.defaultSFA import Model
from tools.pruningSFA import PruningModel
import argparse
import os
from tqdm import tqdm
from torch import nn
from torch import optim
import numpy as np
from tensorboardX import SummaryWriter


from tools import utils


modelName= 'pruneSFA'

parser = argparse.ArgumentParser()
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--epoch', default=1, type=int, help='train epochs')
parser.add_argument('--dataset', default='SHB', type=str, help='dataset')
parser.add_argument('--data_path', default='/home/osk/datasets/ShanghaiTech_Crowd_Counting_Dataset/', type=str, help='path to dataset')
parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--load', default=False, action='store_true', help='load checkpoint')
parser.add_argument('--save_path', default='../checkpoint', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
parser.add_argument('--log', default='../logs', type=str, help='path to log')

args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()


# load dataset
train_dataset = Dataset(args.data_path, args.dataset, True)
train_loader = data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
test_dataset = Dataset(args.data_path, args.dataset, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda:' + str(args.gpu))

# load model
model = PruningModel().cuda()
# model = Model()
writer = SummaryWriter(os.path.join(args.log + '/'+ modelName +'/', 'prune'))

optimizer = optim.Adam(model.parameters(), lr=1e-5)
mseloss = nn.MSELoss(reduction='sum').to(device)
bceloss = nn.BCELoss(reduction='sum').to(device)

if args.load:
    checkpoint = torch.load(os.path.join(args.save_path + '/' + modelName, 'checkpoint_latest.pth'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    best_mae = torch.load(os.path.join(args.save_path + '/' + modelName, 'checkpoint_best.pth'))['mae']
    start_epoch = checkpoint['epoch'] + 1

else:
    best_mae = 999999
    start_epoch = 0

# utils.print_model_parameters(model)

# ################################################
# total = 0
# for m in model.modules():
#     if isinstance(m, nn.BatchNorm2d):
#         total += m.weight.data.shape[0]
#
# bn = torch.zeros(total)
# index = 0
# for m in model.modules():
#     if isinstance(m, nn.BatchNorm2d):
#         size = m.weight.data.shape[0]
#         bn[index:(index+size)] = m.weight.data.abs().clone()
#         index += size
#
# y, i = torch.sort(bn)
# thre_index = int(total * args.percent)
# thre = y[thre_index]
#
# pruned = 0
# cfg = []
# cfg_mask = []
# for k, m in enumerate(model.modules()):
#     # print(m)
#     # test = m
#     if isinstance(m, nn.BatchNorm2d):
#         weight_copy = m.weight.data.abs().clone()
#         # weight_copy.cuda()
#         mask = weight_copy.gt(thre).float()#.cuda()
#         pruned = pruned + mask.shape[0] - torch.sum(mask)
#         m.weight.data.mul_(mask)
#         m.bias.data.mul_(mask)
#         cfg.append(int(torch.sum(mask)))
#         cfg_mask.append(mask.clone())
#         print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
#             format(k, mask.shape[0], int(torch.sum(mask))))
#     elif isinstance(m, nn.MaxPool2d):
#         # print('test')
#         print(m)
#         cfg.append('M')
#
# pruned_ratio = pruned/total
#
# print('Pre-processing Successful!')
# print(pruned_ratio)
# print(cfg)
# utils.print_model_parameters(model)
###############################################

# Make real prune
# print(cfg[0:15])
# ['M', 12, 17, 39, 54, 49, 26, 71, 46, 42, 39, 154, 123, 510, 256, 256, 128, 128, 64, 64, 32, 256, 256, 128, 128, 64, 64, 32, 1, 1]
# newmodel = PruningModel(cfg=cfg[0:13])

# simple test model after Pre-processing prune (simple set BN scales to zeros)
# def test(model):
#     kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#     model.cuda()
#     model.eval()
#
#     for epoch in range(start_epoch, start_epoch + args.epoch):
#         with torch.no_grad():
#             mae, mse = 0.0, 0.0
#             for i, (images, gt) in enumerate(test_loader):
#                 images = images.cuda()
#
#                 predict, _ = model(images)
#
#                 # print('predict:{:.2f} label:{:.2f}'.format(predict.sum().item(), gt.item()))
#                 mae += torch.abs(predict.sum() - gt).item()
#                 mse += ((predict.sum() - gt) ** 2).item()
#
#             mae /= len(test_loader)
#             mse /= len(test_loader)
#             mse = mse ** 0.5
#             print('Epoch:', epoch, 'MAE:', mae, 'MSE:', mse)
        # correct = 0
    # for data, target in test_loader:
    #     if args.cuda:
    #         data, target = data.cuda(), target.cuda()
    #     data, target = Variable(data, volatile=True), Variable(target)
    #     output = model(data)
    #     pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    #     correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    #
    # print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
    #     correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    # return correct / float(len(test_loader.dataset))

# acc = test(model)

#############################
# print_model_parameters(model)

############################
# Param name                     Shape                          Type
# ----------------------------------------------------------------------
# vgg.conv1_1.conv.weight        torch.Size([64, 3, 3, 3])      torch.float32
# vgg.conv1_1.conv.bias          torch.Size([64])               torch.float32
# vgg.conv1_1.bn.weight          torch.Size([64])               torch.float32
# vgg.conv1_1.bn.bias            torch.Size([64])               torch.float32
# vgg.conv1_2.conv.weight        torch.Size([64, 64, 3, 3])     torch.float32
# vgg.conv1_2.conv.bias          torch.Size([64])               torch.float32
# vgg.conv1_2.bn.weight          torch.Size([64])               torch.float32
# vgg.conv1_2.bn.bias            torch.Size([64])               torch.float32
# vgg.conv2_1.conv.weight        torch.Size([128, 64, 3, 3])    torch.float32
# vgg.conv2_1.conv.bias          torch.Size([128])              torch.float32
# vgg.conv2_1.bn.weight          torch.Size([128])              torch.float32
# vgg.conv2_1.bn.bias            torch.Size([128])              torch.float32
# vgg.conv2_2.conv.weight        torch.Size([128, 128, 3, 3])   torch.float32
# vgg.conv2_2.conv.bias          torch.Size([128])              torch.float32
# vgg.conv2_2.bn.weight          torch.Size([128])              torch.float32
# vgg.conv2_2.bn.bias            torch.Size([128])              torch.float32
# vgg.conv3_1.conv.weight        torch.Size([256, 128, 3, 3])   torch.float32
# vgg.conv3_1.conv.bias          torch.Size([256])              torch.float32
# vgg.conv3_1.bn.weight          torch.Size([256])              torch.float32
# vgg.conv3_1.bn.bias            torch.Size([256])              torch.float32
# vgg.conv3_2.conv.weight        torch.Size([256, 256, 3, 3])   torch.float32
# vgg.conv3_2.conv.bias          torch.Size([256])              torch.float32
# vgg.conv3_2.bn.weight          torch.Size([256])              torch.float32
# vgg.conv3_2.bn.bias            torch.Size([256])              torch.float32
# vgg.conv3_3.conv.weight        torch.Size([256, 256, 3, 3])   torch.float32
# vgg.conv3_3.conv.bias          torch.Size([256])              torch.float32
# vgg.conv3_3.bn.weight          torch.Size([256])              torch.float32
# vgg.conv3_3.bn.bias            torch.Size([256])              torch.float32
# vgg.conv4_1.conv.weight        torch.Size([512, 256, 3, 3])   torch.float32
# vgg.conv4_1.conv.bias          torch.Size([512])              torch.float32
# vgg.conv4_1.bn.weight          torch.Size([512])              torch.float32
# vgg.conv4_1.bn.bias            torch.Size([512])              torch.float32
# vgg.conv4_2.conv.weight        torch.Size([512, 512, 3, 3])   torch.float32
# vgg.conv4_2.conv.bias          torch.Size([512])              torch.float32
# vgg.conv4_2.bn.weight          torch.Size([512])              torch.float32
# vgg.conv4_2.bn.bias            torch.Size([512])              torch.float32
# vgg.conv4_3.conv.weight        torch.Size([512, 512, 3, 3])   torch.float32
# vgg.conv4_3.conv.bias          torch.Size([512])              torch.float32
# vgg.conv4_3.bn.weight          torch.Size([512])              torch.float32
# vgg.conv4_3.bn.bias            torch.Size([512])              torch.float32
# vgg.conv5_1.conv.weight        torch.Size([512, 512, 3, 3])   torch.float32
# vgg.conv5_1.conv.bias          torch.Size([512])              torch.float32
# vgg.conv5_1.bn.weight          torch.Size([512])              torch.float32
# vgg.conv5_1.bn.bias            torch.Size([512])              torch.float32
# vgg.conv5_2.conv.weight        torch.Size([512, 512, 3, 3])   torch.float32
# vgg.conv5_2.conv.bias          torch.Size([512])              torch.float32
# vgg.conv5_2.bn.weight          torch.Size([512])              torch.float32
# vgg.conv5_2.bn.bias            torch.Size([512])              torch.float32
# vgg.conv5_3.conv.weight        torch.Size([512, 512, 3, 3])   torch.float32
# vgg.conv5_3.conv.bias          torch.Size([512])              torch.float32
# vgg.conv5_3.bn.weight          torch.Size([512])              torch.float32
# vgg.conv5_3.bn.bias            torch.Size([512])              torch.float32
# amp.conv1.conv.weight          torch.Size([256, 1024, 1, 1])  torch.float32
# amp.conv1.conv.bias            torch.Size([256])              torch.float32
# amp.conv1.bn.weight            torch.Size([256])              torch.float32
# amp.conv1.bn.bias              torch.Size([256])              torch.float32
# amp.conv2.conv.weight          torch.Size([256, 256, 3, 3])   torch.float32
# amp.conv2.conv.bias            torch.Size([256])              torch.float32
# amp.conv2.bn.weight            torch.Size([256])              torch.float32
# amp.conv2.bn.bias              torch.Size([256])              torch.float32
# amp.conv3.conv.weight          torch.Size([128, 512, 1, 1])   torch.float32
# amp.conv3.conv.bias            torch.Size([128])              torch.float32
# amp.conv3.bn.weight            torch.Size([128])              torch.float32
# amp.conv3.bn.bias              torch.Size([128])              torch.float32
# amp.conv4.conv.weight          torch.Size([128, 128, 3, 3])   torch.float32
# amp.conv4.conv.bias            torch.Size([128])              torch.float32
# amp.conv4.bn.weight            torch.Size([128])              torch.float32
# amp.conv4.bn.bias              torch.Size([128])              torch.float32
# amp.conv5.conv.weight          torch.Size([64, 256, 1, 1])    torch.float32
# amp.conv5.conv.bias            torch.Size([64])               torch.float32
# amp.conv5.bn.weight            torch.Size([64])               torch.float32
# amp.conv5.bn.bias              torch.Size([64])               torch.float32
# amp.conv6.conv.weight          torch.Size([64, 64, 3, 3])     torch.float32
# amp.conv6.conv.bias            torch.Size([64])               torch.float32
# amp.conv6.bn.weight            torch.Size([64])               torch.float32
# amp.conv6.bn.bias              torch.Size([64])               torch.float32
# amp.conv7.conv.weight          torch.Size([32, 64, 3, 3])     torch.float32
# amp.conv7.conv.bias            torch.Size([32])               torch.float32
# amp.conv7.bn.weight            torch.Size([32])               torch.float32
# amp.conv7.bn.bias              torch.Size([32])               torch.float32
# dmp.conv1.conv.weight          torch.Size([256, 1024, 1, 1])  torch.float32
# dmp.conv1.conv.bias            torch.Size([256])              torch.float32
# dmp.conv1.bn.weight            torch.Size([256])              torch.float32
# dmp.conv1.bn.bias              torch.Size([256])              torch.float32
# dmp.conv2.conv.weight          torch.Size([256, 256, 3, 3])   torch.float32
# dmp.conv2.conv.bias            torch.Size([256])              torch.float32
# dmp.conv2.bn.weight            torch.Size([256])              torch.float32
# dmp.conv2.bn.bias              torch.Size([256])              torch.float32
# dmp.conv3.conv.weight          torch.Size([128, 512, 1, 1])   torch.float32
# dmp.conv3.conv.bias            torch.Size([128])              torch.float32
# dmp.conv3.bn.weight            torch.Size([128])              torch.float32
# dmp.conv3.bn.bias              torch.Size([128])              torch.float32
# dmp.conv4.conv.weight          torch.Size([128, 128, 3, 3])   torch.float32
# dmp.conv4.conv.bias            torch.Size([128])              torch.float32
# dmp.conv4.bn.weight            torch.Size([128])              torch.float32
# dmp.conv4.bn.bias              torch.Size([128])              torch.float32
# dmp.conv5.conv.weight          torch.Size([64, 256, 1, 1])    torch.float32
# dmp.conv5.conv.bias            torch.Size([64])               torch.float32
# dmp.conv5.bn.weight            torch.Size([64])               torch.float32
# dmp.conv5.bn.bias              torch.Size([64])               torch.float32
# dmp.conv6.conv.weight          torch.Size([64, 64, 3, 3])     torch.float32
# dmp.conv6.conv.bias            torch.Size([64])               torch.float32
# dmp.conv6.bn.weight            torch.Size([64])               torch.float32
# dmp.conv6.bn.bias              torch.Size([64])               torch.float32
# dmp.conv7.conv.weight          torch.Size([32, 64, 3, 3])     torch.float32
# dmp.conv7.conv.bias            torch.Size([32])               torch.float32
# dmp.conv7.bn.weight            torch.Size([32])               torch.float32
# dmp.conv7.bn.bias              torch.Size([32])               torch.float32
# conv_att.conv.weight           torch.Size([1, 32, 1, 1])      torch.float32
# conv_att.conv.bias             torch.Size([1])                torch.float32
# conv_att.bn.weight             torch.Size([1])                torch.float32
# conv_att.bn.bias               torch.Size([1])                torch.float32
# conv_out.conv.weight           torch.Size([1, 32, 1, 1])      torch.float32
# conv_out.conv.bias             torch.Size([1])                torch.float32
# conv_out.bn.weight             torch.Size([1])                torch.float32
# conv_out.bn.bias               torch.Size([1])                torch.float32

#############################

# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# mseloss = nn.MSELoss(reduction='sum').to(device)
# bceloss = nn.BCELoss(reduction='sum').to(device)
#


for epoch in range(start_epoch, start_epoch + args.epoch):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    loss_avg, loss_att_avg = 0.0, 0.0

    for i, (images, density, att) in pbar:
        images = images.to(device)
        density = density.to(device)
        att = att.to(device)
        outputs, attention = model(images)

        # model = huffmancoding.huffman_encode_model(model)

        # print('output:{:.2f} label:{:.2f}'.format(outputs.sum().item() / args.bs, density.sum().item() / args.bs))

        loss = mseloss(outputs, density) / args.bs
        loss_att = bceloss(attention, att) / args.bs * 0.1
        loss_sum = loss + loss_att

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        loss_avg += loss.item()
        loss_att_avg += loss_att.item()

        pbar.set_description("Train: Epoch:{}, Step:{}, Loss:{:.4f} {:.4f}".format(epoch, i, loss_avg / (i + 1), loss_att_avg / (i + 1)))

    writer.add_graph(model,images)
    # writer.add_scalar('loss/train_loss', loss_avg / len(train_loader), epoch)
    # writer.add_scalar('loss/train_loss_att', loss_att_avg / len(train_loader), epoch)

    model.eval()
    with torch.no_grad():
        mae, mse = 0.0, 0.0
        for i, (image_Path,images, gt) in enumerate(test_loader):
            images = images.to(device)

            predict, _ = model(images)

            # print('predict:{:.2f} label:{:.2f}'.format(predict.sum().item(), gt.item()))
            mae += torch.abs(predict.sum() - gt).item()
            mse += ((predict.sum() - gt) ** 2).item()

        mae /= len(test_loader)
        mse /= len(test_loader)
        mse = mse ** 0.5
        print('Epoch:', epoch, 'MAE:', mae, 'MSE:', mse)
        writer.add_scalar('eval/MAE', mae, epoch)
        writer.add_scalar('eval/MSE', mse, epoch)

        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'mae': mae,
                 'mse': mse}
        torch.save(state, os.path.join(args.save_path + '/' + modelName, 'checkpoint_latest.pth'))

        if mae < best_mae:
            best_mae = mae
            torch.save(state, os.path.join(args.save_path + '/' + modelName, 'checkpoint_best.pth'))
    model.train()

writer.close()
