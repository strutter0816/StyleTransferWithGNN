import torch
import torch.utils.data as data
from PIL import Image
from pathlib import Path


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
        return len(self.paths)  # 注意：根据你的处理方式，你可能需要调整这里

    def name(self):
        return 'FlatFolderDataset'
    
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

# 假设FlatFolderDataset类的定义已经存在

# 使用你定义的train_transform
transform = train_transform()

# 创建数据集实例
root = '/root/autodl-tmp/StyleTransferGNN.PyTorch/VideoDataset/content/1'  # 请替换为实际图片文件夹路径
dataset = FlatFolderDataset(root, transform)
img1, next_img1 = dataset.__getitem__(0)
print(img1.shape,next_img1.shape)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 测试读取并显示数据集中的前N个样本
N = 5
for i, (img1, img2) in enumerate(dataloader):
    if i >= N: break  # 只显示前N个样本

    # 将Tensor转换回图片
    img1 = img1.squeeze().permute(1, 2, 0).numpy()
    img2 = img2.squeeze().permute(1, 2, 0).numpy()

    # 创建显示图
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img1)
    ax[0].set_title(f"Image at Index: {i}")
    ax[0].axis('off')  # 关闭坐标轴显示

    ax[1].imshow(img2)
    ax[1].set_title(f"Next Image at Index: {i+1} (or wrapped to 0)")
    ax[1].axis('off')  # 关闭坐标轴显示

    plt.show()
