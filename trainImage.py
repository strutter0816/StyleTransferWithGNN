import argparse
from pathlib import Path
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


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='../datasets/train2014',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='../datasets/train',
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
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
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--graph_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=5000)
parser.add_argument('--log_interval', type=int, default=5000)
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

network = net.GraphNet(encoder, decoder, k=args.k, patch_size=args.patch_size, stride=args.stride, conv_type=args.conv) #if args.graph else net.Net(encoder, decoder)
network.train()

network.to(device)


content_tf = train_transform()
style_tf = train_transform()

content_testtf = test_transform(size=(512, 512))
style_testtf = test_transform(size=(512, 512))

print('\nLoading Content Dataset from ' + args.content_dir + ' ...')
content_dataset = FlatFolderDataset(args.content_dir, content_tf)
print('Loading Style Dataset from ' + args.style_dir + ' ...')
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))


optimizer = torch.optim.Adam(network.update_layers.parameters(), lr=args.lr)

loss_c_sum, loss_s_sum, loss_g_sum = 0.,0.,0.
print('\n{}   Start Training ...'.format(time.ctime()))
for i in range(args.max_iter):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    # print(content_images)
    style_images = next(style_iter).to(device)
    _, loss_c, loss_s = network(content_images, style_images)
    # loss_g = torch.Tensor([0.]).to(device)
    loss = args.content_weight * loss_c + args.style_weight * loss_s 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_c_sum += args.content_weight * loss_c
    loss_s_sum += args.style_weight * loss_s
    writer.add_scalar('loss/Content_loss',loss_c_sum/(i+1),(i+1))
    writer.add_scalar('loss/Style_loss',loss_s_sum/(i+1),(i+1))
    print('\r{}   Batch [{:6d}/{:6d}]   Content Loss: {:3.4f}  Style Loss: {:3.4f}'.format(time.ctime(), i, args.max_iter, loss_c_sum/(i+1), loss_s_sum/(i+1), end='', flush=True))

    if (i+1) % args.save_model_interval == 0 or i+1 == args.max_iter or i==0:
        state_dict = network.update_layers.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, os.path.join(save_ckpt_dir, 'iter_'+str(i+1).zfill(6)+'.pth.tar'))
        torch.save(optimizer.state_dict(), os.path.join(save_ckpt_dir, 'optimizer_iter_'+str(i+1).zfill(6)+'.pth.tar'))

        if i+1 == args.max_iter:
             torch.save(state_dict, os.path.join(save_dir, 'best_model.pth.tar'))
             torch.save(optimizer.state_dict(), os.path.join(save_dir, 'best_optimizer.pth.tar'))
        
        content = content_testtf(Image.open('./content_image/1.jpg').convert("RGB"))
        style = style_testtf(Image.open('./style_image/1.jpg').convert("RGB"))
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

