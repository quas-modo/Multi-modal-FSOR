import torch
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def draw_CAM(model, img_path, save_path, visual_heatmap=False):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).cuda()

    model.eval()

    # 获取模型的最后一个卷积层和全连接层的权重
    final_conv = model.layer4
    fc_params = list(model.fc.parameters())
    weight_softmax = np.squeeze(fc_params[0].data.cpu().numpy())

    # 注册hook提取特征图
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    hook = final_conv.register_forward_hook(hook_feature)

    # 获取模型输出
    output = model(img_tensor)
    output = torch.nn.functional.softmax(output, dim=1)
    pred = torch.argmax(output).item()

    # 移除hook
    hook.remove()

    # 计算CAM
    CAMs = returnCAM(features_blobs[-1], weight_softmax, pred)

    # 从CAM生成热力图并保存
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    heatmap = cv2.resize(CAMs, (width, height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(save_path, superimposed_img)

    if visual_heatmap:
        plt.imshow(superimposed_img)
        plt.axis('off')
        plt.show()

def returnCAM(feature_conv, weight_softmax, class_idx):
    # 确定特征图的形状
    # 假设 batch_size 总是 1，这是通常的情况，尤其是在单张图像处理中
    _, nc, h, w = feature_conv.shape  # 添加 _ 来接收批次维度

    # 计算 CAM
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = np.maximum(cam, 0)  # 仅保留正激活
    cam_img = cam / np.max(cam)  # 归一化处理
    return cv2.resize(cam_img, (w, h))  # 调整大小以匹配输入图像尺寸

# def returnCAM(feature_conv, weight_softmax, class_idx):
#     _, nc, h, w = feature_conv.shape
#
#     # 直接对特征图进行平均，而不是加权平均
#     cam = np.mean(feature_conv, axis=1)[0]  # 假设批大小为1，平均所有通道
#
#     # 对结果进行非线性变换可以增强视觉效果
#     cam = np.maximum(cam, 0)
#     cam = np.sqrt(cam)  # 使用平方根可以增强低值区域的可见度
#     cam = np.sqrt(cam)
#     # cam = np.sqrt(cam)
#     # cam = np.sqrt(cam)
#     print(cam)
#     cam = cam - np.min(cam)
#     # cam = cam - np.min(cam)
#     cam_img = cam / np.max(cam)
#     cam_img = 1 - cam_img
#     return cv2.resize(cam_img, (w, h))




# 加载预训练模型并指定设备
model = models.resnet50(pretrained=True).cuda()
draw_CAM(model, "car.jpg", "heatmap_car.jpg")
