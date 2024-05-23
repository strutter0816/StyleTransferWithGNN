import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net, graph_agg_v2
from function import adaptive_instance_normalization, coral
import utils


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def test_transform_content(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.Pad(padding=32, padding_mode='reflect'))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--content', type=str)
parser.add_argument('--content_dir', type=str, default='./content')
parser.add_argument('--style', type=str)
parser.add_argument('--style_dir', type=str, default='./style')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--model', type=str, default='./experiments/checkpoints/best_model.pth.tar')
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--conv', type=str, default='gatconv')

# Additional options
parser.add_argument('--content_size', type=int, default=512)
parser.add_argument('--style_size', type=int, default=512)
parser.add_argument('--crop', type=bool, default=True)
# parser.add_argument('--crop', action='store_true')
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--output_name', type=str, default='output.jpg')

parser.add_argument('--save_each', action='store_true')
# parser.add_argument('--graph', action='store_true')

# Advanced options
parser.add_argument('--preserve_color', action='store_true')
parser.add_argument('--alpha', type=float, default=1.0)

args = parser.parse_args()
device = torch.device("cuda:" + str(args.cuda))

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
# if args.save_each:
output_dir_each = Path(os.path.join(args.output_dir, 'images'))
output_dir_each.mkdir(exist_ok=True, parents=True)
print(output_dir_each)
#     output_dir_each = args.output_dir
# print(output_dir_each)
# output_dir_each.mkdir(exist_ok=True, parents=True)


# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = [Path(args.style)]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

encoder = net.vgg
decoder = net.decoder
print("pre model loading")
encoder.load_state_dict(torch.load(args.vgg))
encoder = nn.Sequential(*list(encoder.children())[:31])
print("pre model loaded")
# network = net.GraphNet_v0(encoder, decoder, graph)# if args.graph else net.Net(encoder, decoder)
network = net.GraphNet(encoder, decoder, k=args.k, patch_size=args.patch_size, stride=args.stride,
                       conv_type=args.conv)  # if args.graph else net.Net(encoder, decoder)

network.eval()
network.to(device)
print(" model loading")
network.update_layers.load_state_dict(torch.load(args.model))
print(" model loaded")
content_tf = test_transform_content(args.content_size, args.crop)
style_tf = test_transform_content(args.style_size, args.crop)

output_list = []
style_list = []
for content_path in content_paths:
    for style_path in style_paths:
        print(content_path)
        print(style_path)
        # TODO resize 512 function
        content = content_tf(Image.open(str(content_path)).convert('RGB'))
        style = style_tf(Image.open(str(style_path)).convert('RGB'))
        if args.preserve_color:
            style = coral(style, content)
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)

        with torch.no_grad():
            output, _, _ = network(content, style, args.alpha)
            output = transforms.CenterCrop(args.content_size)(output)
            output = output.cpu()
            output_list.append(output)
            style_list.append(style.cpu())
            print("image generated")
            #             if args.save_each:
            save_image(output, os.path.join(output_dir_each,
                                            '{:s}_stylized_{:s}.jpg'.format(content_path.stem, style_path.stem)))

# if args.output_name is not None:
#    utils.save_all_image(output_list, style_list, os.path.join(output_dir, args.output_name))
