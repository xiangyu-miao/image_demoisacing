import cv2
import numpy as np
from glob import glob
import os
from tqdm import tqdm

# 获取项目根目录路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_DIR = os.path.expanduser("~/data")
DEFAULT_INPUT_DIR = os.path.join(DATA_DIR, "img_align_celeba/img_align_celeba")
DEFAULT_MOSAIC_DIR = os.path.join(DATA_DIR, "mosaic")
DEFAULT_ORIGINAL_DIR = os.path.join(DATA_DIR, "original")

def apply_mosaic(image, kernel_size=10):
    """生成马赛克效果"""
    h, w = image.shape[:2]
    image = cv2.resize(image, (w // kernel_size, h // kernel_size), interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
    return image

def prepare_data(input_dir=DEFAULT_INPUT_DIR, mosaic_dir=DEFAULT_MOSAIC_DIR, original_dir=DEFAULT_ORIGINAL_DIR, kernel_size=10):
    """生成马赛克图像并保存"""
    # 创建输出目录
    os.makedirs(mosaic_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)

    # 获取图像文件路径列表
    img_paths = glob(os.path.join(input_dir, "*.jpg"))
    
    # 使用 tqdm 显示进度条
    for img_path in tqdm(img_paths, desc="Processing images", unit="image"):
        image = cv2.imread(img_path)
        mosaic_image = apply_mosaic(image, kernel_size)
        
        # 保存马赛克图像和清晰图像
        base_name = os.path.basename(img_path)  # 获取文件名
        cv2.imwrite(os.path.join(mosaic_dir, f"mosaic_{base_name}"), mosaic_image)
        cv2.imwrite(os.path.join(original_dir, f"original_{base_name}"), image)
    
    print("Data preparation completed.")

if __name__ == "__main__":
    prepare_data()