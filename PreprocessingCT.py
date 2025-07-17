
"""
@Project : Preprocess
@File : preprocess.py
@Author : PengYuan
@Date : 17/05/2023 19:23
@Brief : preprocess
"""
#############
#!/home/PengYaun/anaconda3 python
# -*- coding: UTF-8 -*-
"""
@Project : Preprocess 
@File : utils.py
@Author : PengYuan    
@Date : 14/05/2023 19:39 
@Brief : Resample by SimpelItK
"""
import SimpleITK as sitk
import numpy as np
import copy

def itkResampleBySpacing(itkimage, newSpacing, resamplemethod=sitk.sitkLinear):
    resampler = sitk.ResampleImageFilter()
    originSpacing = itkimage.GetSpacing()
    originSize = itkimage.GetSize()
    ratio = originSpacing / newSpacing
    newSize = np.round(originSize * ratio)
    newSize = newSize.astype('int32')
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled, ratio

def itkStandardization(itkimage):
    itkimage = sitk.Cast(itkimage, sitk.sitkFloat32)
    data = sitk.GetArrayFromImage(itkimage)
    mu = np.mean(data)
    sigma = np.std(data)
    data = (data - mu) / sigma
    max = np.max(data)
    min = np.min(data)
    itkimage = (itkimage - mu) / sigma
    itkimage = itkimage / (max - min) * 4095
    itkimage = sitk.Cast(itkimage, sitk.sitkInt16)
    return itkimage

def itkNormalization(itkimage):
    itkimage = sitk.Cast(itkimage, sitk.sitkFloat32)
    data = sitk.GetArrayFromImage(itkimage)
    max = np.max(data)
    min = np.min(data)
    itkimage = (itkimage - min) / (max - min)
    return itkimage

def itkWindowTransform(itkimage, WW, WL):
    data = sitk.GetArrayFromImage(itkimage)
    minWindow = int(float(WL) - float(WW) / 2.0)
    maxWindow = minWindow + WW - 1
    data[data < minWindow] = minWindow
    data[data > maxWindow] = maxWindow
    itkimage_WT = sitk.GetImageFromArray(data)
    itkimage_WT.SetSpacing(itkimage.GetSpacing())
    return itkimage_WT

#def itkPadding(itkimage, padding_size, value=0):
def itkPadding(itkimage, padding_size, value=-1000):#针对多模态数据，填充-1000
    data = sitk.GetArrayFromImage(itkimage)
    shape = np.array(data.shape)
    # assert (shape <= padding_size).all(), "Padding size should bigger than the image shape"
    padding = [0] * 3
    padding_left = [0] * 3
    padding_size = padding_size[::-1]
    for i in range(3):
        padding[i] = max(0, padding_size[i] - shape[i])
        padding_left[i] = max(0, padding[i] // 2)
    data_padding = np.pad(data, ((padding_left[0], padding[0] - padding_left[0]),
                                 (padding_left[1], padding[1] - padding_left[1]),
                                 (padding_left[2], padding[2] - padding_left[2])),
                          mode='constant',
                          constant_values=value
                          )
    itkimage_padding = sitk.GetImageFromArray(data_padding)
    itkimage_padding.SetSpacing(itkimage.GetSpacing())
    return itkimage_padding, padding_left[::-1]

def itkCrop(itkimage, patch_size, residual):
    residual = residual[::-1]  # [W, H, D]
    itkimage_padding, padding_left = itkPadding(itkimage, patch_size,-1000)
    dim = np.array(itkimage_padding.GetSize())
    center = dim // 2 + np.array(residual).astype('int')
    bottom_legal = np.array(patch_size) // 2 - dim // 2
    top_legal = dim - np.array(patch_size) // 2 - dim // 2
    print('input_residual: [%.1f, %.1f, %.1f]' % (residual[2], residual[1], residual[0]))
    print('legal_residual_range: [%d ~ %d, %d ~ %d, %d ~ %d]' % (bottom_legal[2], top_legal[2],
                                                                 bottom_legal[1], top_legal[1],
                                                                 bottom_legal[0], top_legal[0],
                                                                 ))
    bottom = center - np.array(patch_size) // 2
    x0, y0, z0 = bottom
    x0 = min(max(x0, 0), dim[0] - patch_size[0])
    y0 = min(max(y0, 0), dim[1] - patch_size[1])
    z0 = min(max(z0, 0), dim[2] - patch_size[2])
    crop_left = [x0, y0, z0]
    return itkimage_padding[x0:x0 + patch_size[0], y0:y0 + patch_size[1],
           z0:z0 + patch_size[2]], crop_left, padding_left

def itkCut(itkimage, patch_size=[96, 96, 96], overlap_size=[0, 0, 0]):
    itkimage_padding, _ = itkPadding(itkimage, patch_size,-1000)
    width, height, depth = itkimage_padding.GetSize()
    patch_width, patch_height, patch_depth = patch_size
    overlap_width, overlap_height, overlap_depth = overlap_size
    z_step = patch_depth - overlap_depth
    y_step = patch_height - overlap_height
    x_step = patch_width - overlap_width
    patches = []
    for z in range(0, depth + 1, z_step):
        if z + z_step > depth:
            z = depth - z_step
        for y in range(0, height + 1, y_step):
            if y + y_step > height:
                y = height - y_step
            for x in range(0, width + 1, x_step):
                if x + x_step > width:
                    x = width - x_step
                patch = copy.deepcopy(itkimage_padding[x:x + patch_width, y:y + patch_height, z:z + patch_depth])
                patches.append(patch)
    return patches
#############

import os
import re
import numpy as np
import SimpleITK as sitk
from utils import *
def normalize(image, vmin=-1024, vmax=3072):
    image = (image - vmin) / (vmax - vmin)
    image[image < 0] = 0
    image[image > 1] = 1
    return image
import os
def set_origin_to_zero(image, new_origin=(0.0, 0.0, 0.0)):
    '''
    图像Origin归零，但保持信息在物理空间不变
    '''
    # 获取当前原点和间隔
    current_origin = image.GetOrigin()
    spacing = image.GetSpacing()

    # 计算新的原点
    # new_origin = (0.0, 0.0, 0.0)

    # 计算平移向量
    translation_vector = [new_origin[i] - current_origin[i] for i in range(3)]

    # 应用平移变换
    translation_transform = sitk.TranslationTransform(3, translation_vector)
    image = sitk.Resample(image, image.GetSize(), translation_transform, sitk.sitkLinear, current_origin, spacing)

    # 设置新的原点
    image.SetOrigin(new_origin)
    return image
folder_path = r"/mnt/d/work4/OnlyLung/train"
fileStore= r"/mnt/d/work4/OnlyLung/trainResample"
files = os.listdir(folder_path)
# 打印所有文件和文件夹的名称
for file in files:
    # 使用绝对路径
    file_path = os.path.join(folder_path, file)
    # 判断是否为文件
    if os.path.isfile(file_path):
        print(f"File: {file_path}")
        #####################
        image_std = sitk.ReadImage(file_path)
        imageOrigin = image_std.GetOrigin()
        direction = image_std.GetDirection()

        dst_scale = [2.0,2.0,2.0]
        image_std, ratio = itkResampleBySpacing(image_std, np.array(dst_scale), resamplemethod=sitk.sitkLinear)
        #image_std = set_origin_to_zero(image_std)
        dst_dim = [160,160,160]
        residual = [0,0,0]
       # image_crop = image_std
       # imageOrigin2[:] = imageOrigin2[:].astype('float') * ratio
        # 4. crop
        ###########################
        image_crop, crop_left, padding_left = itkCrop(image_std, dst_dim, residual)
        image_crop = sitk.Cast(image_crop, sitk.sitkFloat32)
        # imageOrigin2[:] -= np.array(crop_left)
        # imageOrigin2[:] += np.array(padding_left)
        #image_crop = image_std
        ############################

        temp = fileStore+"//"+os.path.basename(file_path)
        image_crop = sitk.GetArrayFromImage(image_crop)
        print("输出最大最小:",np.max(image_crop),np.min(image_crop))
        #image_crop = normalize(image_crop, np.min(image_crop),np.max(image_crop))
        image_crop = sitk.GetImageFromArray(image_crop)
        image_crop.SetSpacing(dst_scale)
        image_crop.SetDirection(direction)
        ########################

       # imageOrigin2 = np.array(imageOrigin)
        #######################
        #image_crop.SetOrigin(imageOrigin2)
        sitk.WriteImage(image_crop, temp)
        #######################
    # 判断是否为文件夹
    elif os.path.isdir(file_path):
        print(f"Folder: {file_path}")
