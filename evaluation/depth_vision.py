import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

def mask_corners(data, mask_size=22):
    # 创建一个全为True的掩码数组
    mask = np.ones(data.shape, dtype=bool)
    
    # 设置四个角的掩码为False
    mask[:mask_size, :mask_size] = False  # 左上角
    mask[-mask_size:, :mask_size] = False  # 左下角
    mask[:mask_size, -mask_size:] = False  # 右上角
    mask[-mask_size:, -mask_size:] = False  # 右下角
    
    # 应用掩码
    masked_data = np.ma.masked_array(data, ~mask)
    return masked_data

def normalize_depth_map(depth_map):
    """
    归一化深度图，将深度值从[min, max]映射到[0, 1]
    """
    mask_depth_map = mask_corners(depth_map)
    min_val = np.min(mask_depth_map)
    max_val = np.max(mask_depth_map)
    
    # min_val, max_val = np.percentile(depth_map, [0, 99.5])  # 99.9分位数，排除角落的离群点
    normalized_depth_map = (depth_map - min_val) / (max_val - min_val)
    normalized_depth_map[normalized_depth_map < 0] = 0
    normalized_depth_map[normalized_depth_map > 1] = 1
    
    return normalized_depth_map

def depth_map_to_pixel(depth_map):
    """
    将归一化后的深度图映射到[0, 255]的像素值
    """
    # output = output.permute(0, 2, 3, 1)
    # output = 255 * 1 / (1 + torch.exp(-output))
    normalized_depth_map = normalize_depth_map(depth_map)
    pixel_depth_map = (normalized_depth_map * 255).astype(np.uint8)
    return pixel_depth_map

def visualize_depth_map(pixel_depth_map):
    """
    可视化深度图
    """
    plt.imsave('/18179377869/code/Multi-Task-Transformer/yxz/depth.png', pixel_depth_map, cmap='gray')
 
def depth_save(pixel_depth_map,outdir):
    pixel_depth_map = depth_map_to_pixel(pixel_depth_map)

    im_color = cv2.applyColorMap(pixel_depth_map, cv2.COLORMAP_JET) 
    #convert to mat png
    im=Image.fromarray(im_color)
    #save image
    # im.save(os.path.join(outdir,os.path.basename(pngfile)))
    im.save(outdir)

    

# 示例使用
if __name__ == "__main__":
    # 创建一个模拟的深度图
    # depth_map = np.random.rand(480, 640) * 100  # 480x640 的深度图，深度值范围在0到100之间
    import scipy.io as scio
    depth_map = scio.loadmat('/18179377869/code/Multi-Task-Transformer/yxz/work_dir/nyud_vitLp16_vision/results/depth/0001.mat')['depth']

    depth_save(depth_map,"/18179377869/code/Multi-Task-Transformer/yxz/depth.png")#C:/Users/BAMBOO/Desktop/source pics/rgbd_6/color
    # visualize_depth_map(pixel_depth_map)
