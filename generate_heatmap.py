import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 根据你提供的绝对路径，从 deep_sort 模块中准确导入 Net
from deep_sort.deep_sort.deep.model import Net


def main():
    # ========================== 配置区 ==========================
    # 1. 准备好你的测试图片 (请确保当前目录下有这张图)
    test_image_path = "test_person.jpg"

    # 2. 权重文件的相对路径 (通常在你项目里的这个位置)
    weight_path = "deep_sort/deep_sort/deep/checkpoint/ckpt.t7"

    # 3. 输出图片的名字
    output_name = "heatmap_result.jpg"
    # ============================================================

    if not os.path.exists(test_image_path):
        print(f"[报错] 找不到图片 {test_image_path}，请先放一张测试图片到当前目录！")
        return

    if not os.path.exists(weight_path):
        print(f"[报错] 找不到权重文件 {weight_path}，请检查路径！")
        return

    print("[*] 正在加载网络模型...")
    model = Net(num_classes=751)

    # 增加 weights_only=False 消除警告
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)

    # 精准提取你报错里提示的 net_dict
    if 'net_dict' in checkpoint:
        model.load_state_dict(checkpoint['net_dict'])
    elif 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print("[*] 正在预处理图像...")
    img = Image.open(test_image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 64)),  # 🎯 修复：PyTorch transforms 是 (高, 宽)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)

    rgb_img = cv2.imread(test_image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (64, 128))  # 🎯 修复：OpenCV resize 是 (宽, 高)
    rgb_img = np.float32(rgb_img) / 255

    print("[*] 正在生成 Grad-CAM 热力图...")
    # 🎯 靶向层设置：提取 layer4 最后一个 BasicBlock 的特征
    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_name, visualization)
    print(f"[+] 成功！热力图已保存为: {output_name}")


if __name__ == '__main__':
    main()