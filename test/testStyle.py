import torch
from PIL import Image
from torchvision import transforms


# 定义了您的训练变换
def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

# 应用定义的变换
# TODO 弄懂输入参数数量问题
transform = train_transform()
batch_size = 8
# 假设style_image_path是风格图像的文件路径
style_image = Image.open('/root/autodl-tmp/StyleTransferGNN.PyTorch/style_image/1.jpg').convert('RGB')

# 应用变换
style_tensor = transform(style_image)  # 添加batch维度
# 复制style_tensor N次，生成具有批次大小的风格张量
style_batch = torch.stack([style_tensor] * batch_size)
# 如果CUDA可用，将风格张量移至GPU
if torch.cuda.is_available():
    style_batch = style_batch.to('cuda')

# 创建一个 "infinite" style tensor generator
# def get_infinite_style():
#     while True:
#         yield style_tensor


# 无限生成指定批次大小的风格张量
def get_infinite_style(style_batch):
    while True:
        yield style_batch

# 调用无限生成器函数并传递批次张量
style_iter = get_infinite_style(style_batch)
image = next(style_iter)
print(image.shape)

# 注意：剩下的内容加载代码保持不变，根据需要调整批次大小和num_workers等参数
# content_loader = DataLoader(...)


