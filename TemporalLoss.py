import torch
import cv2
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def opticalflow(img1, img2):
    """
    img1:frame(t-1)
    img2:frame(t)
    """
    img1 = np.squeeze(np.transpose(img1, (0, 2, 3, 1)))
    img2 = np.squeeze(np.transpose(img2, (0, 2, 3, 1)))
    prvs_r = img1[:, :, 0]
    prvs_g = img1[:, :, 1]
    prvs_b = img1[:, :, 2]

    next_r = img2[:, :, 0]
    next_g = img2[:, :, 1]
    next_b = img2[:, :, 2]

    flow_r = cv2.calcOpticalFlowFarneback(prvs_r, next_r, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_g = cv2.calcOpticalFlowFarneback(prvs_g, next_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_b = cv2.calcOpticalFlowFarneback(prvs_b, next_b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    real_flow_r = np.sqrt(np.power(flow_r[..., 0], 2) + np.power(flow_r[..., 1], 2))
    real_flow_g = np.sqrt(np.power(flow_g[..., 0], 2) + np.power(flow_g[..., 1], 2))
    real_flow_b = np.sqrt(np.power(flow_b[..., 0], 2) + np.power(flow_b[..., 1], 2))

    real_flow_rgb = np.asarray([real_flow_r, real_flow_g, real_flow_b])

    real_flow = real_flow_r * 0.299 + real_flow_g * 0.587 + real_flow_b * 0.114

    real_flow = cv2.normalize(real_flow, None, 0, 1, cv2.NORM_MINMAX)

    rgb = torch.from_numpy(np.expand_dims(real_flow_rgb, 0)).float()
    gray = torch.from_numpy(np.expand_dims(np.expand_dims(real_flow,0),0)).float()
    return rgb, gray

class TemporalLoss(nn.Module):
    """
    x: frame t 
    f_x1: optical flow(frame t-1)
    cm: confidence mask of optical flow 
    """
    def __init__(self, gpu):
        super(TemporalLoss, self).__init__()
        self.loss = nn.MSELoss().cuda() if gpu else nn.MSELoss()

    def forward(self, x, f_x1, cm):
        assert x.device == f_x1.device == cm.device, "All inputs must be on the same device"
        _, c, h, w = x.shape
        power_sub = (x - f_x1) ** 2
        loss = torch.sum(cm * power_sub[:, 0, :, :] + cm * power_sub[:, 1, :, :] + cm * power_sub[:, 2, :, :]) / (w * h)
        return loss
if __name__ == "__main__":
    # 设置是否使用GPU
    use_gpu = torch.cuda.is_available()

    # 实例化TemporalLoss
    temporal_loss_fn = TemporalLoss(gpu=use_gpu)

    # 创建两个随机样本图像作为连续帧
    frame_t = torch.rand(8, 3, 256, 256).cuda() if use_gpu else torch.rand(1, 3, 224, 224)
    frame_t_minus_1 = torch.rand(8, 3, 256, 256).cuda() if use_gpu else torch.rand(1, 3, 224, 224)

    # 计算两帧之间的光流
    f_x1_cpu, _ = opticalflow(frame_t_minus_1.cpu().numpy(), frame_t.cpu().numpy()) # 前一帧需要移动到CPU上进行光流计算
    f_x1 = f_x1_cpu.to(frame_t.device) # 确保将计算结果移动到原始帧相同的设备

    # 创建置信度掩码，我们假设光流是完全准确的
    confidence_mask = torch.ones_like(f_x1).to(frame_t.device) # 确保置信度掩码在相同的设备上

    # 计算Temporal Loss
    loss_value = temporal_loss_fn(frame_t, f_x1, confidence_mask)
    print('Temporal loss:', loss_value.item())