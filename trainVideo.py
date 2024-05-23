import argparse
from pathlib import Path
import torch
import cv2
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
import time
# import pandas as pd

from TemporalLoss import TemporalLoss, opticalflow
import net, graph_agg_v2
from sampler import InfiniteSamplerWrapper
import pdb
from torchvision.utils import save_image

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True

writer = SummaryWriter(log_dir='/root/tf-logs')

def test_transform(size, crop=False):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)
# dataset for Vimeo dataset
class VimeoDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = self._get_image_pairs(self.root_dir)

    def _get_image_pairs(self, root_dir):
        image_pairs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if filenames:
                sorted_files = sorted([os.path.join(dirpath, file) for file in filenames if file.endswith(".png")])
                for i in range(len(sorted_files) - 1):
                    image_pairs.append((sorted_files[i], sorted_files[i+1]))
        return image_pairs

    def __getitem__(self, index):
        img_path1, img_path2 = self.image_pairs[index]
        
        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

    def __len__(self):
        return len(self.image_pairs)
    def name(self):
        return 'VimeoDataset'
# 修改dataset为连续输出两个连续帧图像
class FlatFolderFramesDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderFramesDataset, self).__init__()
        self.root = root
        self.paths = sorted(list(Path(self.root).glob('*')))  # 确保路径是排序的
        self.transform = transform

    def __getitem__(self, index):
        # 获取当前帧的路径
        path = self.paths[index]
        # 用于处理到达最末帧时的情况，循环回第一个帧
        next_index = (index + 1) % len(self.paths)  # 循环索引
        next_path = self.paths[next_index]

        # 载入并转换当前帧
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)

        # 载入并转换下一个帧
        next_img = Image.open(str(next_path)).convert('RGB')
        next_img = self.transform(next_img)

        # 返回连续的两个帧
        return img, next_img

    def __len__(self):
        return len(self.paths)  

    def name(self):
        return 'FlatFolderFramesDataset'
class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'
    

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# 设置是否使用GPU
use_gpu = torch.cuda.is_available()

# 实例化TemporalLoss
temporal_loss = TemporalLoss(gpu=use_gpu)



parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='/root/autodl-fs/vimeo_interp_test/input',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='/root/autodl-tmp/StyleTransferGNN.PyTorch/styleForVideo/style1.JPG',
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./models/vgg_normalised.pth')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--conv', type=str, default='gatconv')

# training options 
parser.add_argument('--save_dir', type=str, default='./checkpoints',
                    help='Directory to save the model')
parser.add_argument('--save_img_dir', type=str, default='./results',
                    help='Directory to save the model')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--temporal_weight',type=float,default=10**1.5) # best perform with another 10 iterations
parser.add_argument('--graph_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=2)
parser.add_argument('--save_model_interval', type=int, default=10)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--graph', action='store_true')
args = parser.parse_args()

device = torch.device('cuda:'+str(args.cuda))
save_dir = Path(os.path.join('experiments',args.save_dir))
save_dir.mkdir(exist_ok=True, parents=True)
save_ckpt_dir = Path(os.path.join('experiments',args.save_dir,'checkpoints'))
save_ckpt_dir.mkdir(exist_ok=True, parents=True)
save_csv_path = Path(os.path.join('experiments',args.save_dir,'loss.csv'))

if not os.path.exists(args.save_img_dir):
    os.mkdir(args.save_img_dir)

# models
encoder = net.vgg
decoder = net.decoder

encoder.load_state_dict(torch.load(args.vgg))
encoder = nn.Sequential(*list(encoder.children())[:31])

network = net.GraphNet(encoder, decoder, k=args.k, patch_size=args.patch_size, stride=args.stride, conv_type=args.conv) #if args.graph else net.Net(encoder, decoder)\
network.update_layers.load_state_dict(torch.load("/root/autodl-tmp/StyleTransferGNN.PyTorch/experiments/checkpoints/t_weight_31.622776601683793_best_model.pth.tar"))
print("loaded model trained")
network.train()

network.to(device)
 
content_tf = train_transform()
style_tf = train_transform()

content_testtf = test_transform(size=(512, 512))
style_testtf = test_transform(size=(512, 512))

print('\nLoading Content Dataset from ' + args.content_dir + ' ...')
content_dataset = VimeoDataset(args.content_dir, content_tf)
print('Loading Style Dataset from ' + args.style_dir + ' ...')

class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataiter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataiter)
        except StopIteration:
            # Restart the generator if the previous generator is exhausted.
            self.dataiter = iter(self.dataloader)
            batch = next(self.dataiter)
        return batch
content_dataloader = data.DataLoader(
    content_dataset, batch_size=args.batch_size, 
    shuffle=False,num_workers=args.n_threads)
infinite_dataloader = InfiniteDataLoader(content_dataloader)

content_iter = iter(infinite_dataloader)


# 目前style image 就一个
# 假设style_image_path是风格图像的文件路径
# batch_size大小
batch_size = args.batch_size  # 请把N替换成您需要的批次大小

# 加载和转换风格图像
style_image = Image.open(args.style_dir).convert('RGB')
transform = train_transform()
# 应用变换并增加批次维度
style_tensor = transform(style_image)

# 复制style_tensor N次，生成具有批次大小的风格张量
style_batch = torch.stack([style_tensor] * batch_size)
if torch.cuda.is_available():
    style_batch = style_batch.to('cuda')
# 无限生成指定批次大小的风格张量
def get_infinite_style(style_batch):
    while True:
        yield style_batch
# 调用无限生成器函数并传递批次张量
style_iter = get_infinite_style(style_batch)
                                      
# 数据dataset dataloader准备结束

optimizer_state = torch.load('/root/autodl-tmp/StyleTransferGNN.PyTorch/experiments/checkpoints/t_weight_31.622776601683793_best_optimizer.pth.tar')
optimizer = torch.optim.Adam(network.update_layers.parameters(), lr=args.lr)
optimizer.load_state_dict(optimizer_state)

loss_c_sum, loss_s_sum, loss_g_sum ,loss_t_sum = 0.,0.,0.,0.
print('\n{}   Start Training ...'.format(time.ctime()))

for i in range(args.max_iter): #range(args.max_iter) 
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter)
    # 获得两个连续帧
    frame = content_images[0].to(device)
    next_frame = content_images[1].to(device)
    # print(frame.shape)
    # print(content_images)
    style_images = next(style_iter).to(device)
    # _, loss_c, loss_s = network(content_images, style_images)
    frame_o_fs,frame_loss_c,frame_loss_s = network(frame,style_images) # fame_o_fs   torch.Size([8, 3, 256, 256])
    # print("stylized frame_o_fs",frame_o_fs.shape) 
    next_frame_o_fs,next_frame_loss_c,next_frame_loss_s = network(next_frame,style_images)
    loss_c = frame_loss_c+next_frame_loss_c
    loss_s = frame_loss_s+next_frame_loss_s
    # print(len(frame_o_fs),len(next_frame_o_fs))
    f_x1_cpu,cm_cpu = opticalflow(frame_o_fs.detach().cpu().numpy(),next_frame_o_fs.detach().cpu().numpy()) # CPU 计算
    # 放回到GPU上
    f_x1 = f_x1_cpu.to(frame_o_fs.device)
    cm = cm_cpu.to(frame_o_fs.device)
    loss_t = temporal_loss(next_frame_o_fs,f_x1,cm)
    print("loss_t",loss_t)
    # loss_g = torch.Tensor([0.]).to(device)
    loss = args.content_weight * loss_c + args.style_weight * loss_s + args.temporal_weight *loss_t
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_c_sum += args.content_weight * loss_c
    loss_s_sum += args.style_weight * loss_s
    loss_t_sum += args.temporal_weight*loss_t
    writer.add_scalar(f'loss/Content_loss',loss_c_sum/(i+1),(i+1))
    writer.add_scalar(f'loss/Style_loss',loss_s_sum/(i+1),(i+1))
    writer.add_scalar(f'loss/Temporal_loss',loss_t_sum/(i/1),(i+1))
    
    print('\r{}    Batch [{:6d}/{:6d}]   Content Loss: {:3.4f}  Style Loss: {:3.4f} Temporal Loss:{:3.4f}'.format(time.ctime(), i, args.max_iter, loss_c_sum/(i+1), loss_s_sum/(i+1),loss_t_sum/(i+1), end='', flush=True))

    if (i+1) % args.save_model_interval == 0 or i+1 == args.max_iter or i==0:
        state_dict = network.update_layers.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, os.path.join(save_ckpt_dir, f'iter_'+str(i+1).zfill(6)+'.pth.tar'))
        torch.save(optimizer.state_dict(), os.path.join(save_ckpt_dir, f'optimizer_iter_'+str(i+1).zfill(6)+'.pth.tar'))

        if i+1 == args.max_iter:
            torch.save(state_dict, os.path.join(save_dir, f't_weight_{args.temporal_weight}_best_model.pth.tar'))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, f't_weight_{args.temporal_weight}_best_optimizer.pth.tar'))
        
        content = content_testtf(Image.open('./content_image/1.jpg').convert("RGB"))
        style = style_testtf(Image.open('/root/autodl-tmp/StyleTransferGNN.PyTorch/styleForVideo/style1.JPG').convert("RGB"))
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        network.eval()
        with torch.no_grad():
            output, _, _ = network(content, style, 1.0)
        output = output.cpu()
        save_image(output, os.path.join(args.save_img_dir, 'stylized_{}.jpg'.format(str(i))))
        network.train()
writer.close()
print('\n{}   Finish Training.'.format(time.ctime()))
print('\nTrained model saved at', save_dir, '\n')

