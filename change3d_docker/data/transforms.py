# Copyright (c) Duowang Zhu.
# All rights reserved.

import numpy as np
import torch
import random
import cv2
from typing import List, Tuple, Union, Optional, Sequence, Callable, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 添加可序列化的转换组合类
class ComposedTransforms:
    """可序列化的转换函数组合器"""
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list
    
    def __call__(self, img, label=None):
        for t in self.transforms_list:
            img, label = t(img, label)
        return img, label


# 可序列化的变换类
class ScaleTransform:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
    def __call__(self, img, label):
        img = cv2.resize(img, (self.width, self.height))
        label = cv2.resize(label, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return [img, label]


class ResizeTransform:
    def __init__(self, min_size, max_size=None, strict=False):
        if not isinstance(min_size, (list, tuple)):
            self.min_size = (min_size,)
        else:
            self.min_size = min_size
        self.max_size = max_size
        self.strict = strict
        
    def get_size(self, image_size):
        w, h = image_size
        
        if not self.strict:
            size = random.choice(self.min_size)
            
            if self.max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                
                if max_original_size / min_original_size * size > self.max_size:
                    size = int(round(self.max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)
        else:
            if w < h:
                return (self.max_size, self.min_size[0])
            else:
                return (self.min_size[0], self.max_size)
    
    def __call__(self, image, label):
        size = self.get_size(image.shape[:2])
        image = cv2.resize(image, size)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return (image, label)


class RandomCropResizeTransform:
    def __init__(self, crop_area):
        self.crop_area = crop_area
        
    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.crop_area)
            y1 = random.randint(0, self.crop_area)

            img_crop = img[y1:h-y1, x1:w-x1]
            label_crop = label[y1:h-y1, x1:w-x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)

            return img_crop, label_crop
        return [img, label]


class RandomFlipTransform:
    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.flip(image, 0)  # horizontal flip
            label = cv2.flip(label, 0)
            
        if random.random() < 0.5:
            image = cv2.flip(image, 1)  # vertical flip
            label = cv2.flip(label, 1)
            
        return [image, label]


class BCDRandomExchangeTransform:
    def __call__(self, image, label):
        if random.random() < 0.5:
            pre_img = image[:, :, 0:3]
            post_img = image[:, :, 3:6]
            image = np.concatenate((post_img, pre_img), axis=2)
        return [image, label]


class SCDRandomExchangeTransform:
    def __call__(self, image, label):
        if random.random() < 0.5:
            pre_img = image[:, :, 0:3]
            post_img = image[:, :, 3:6]
            image = np.concatenate((post_img, pre_img), axis=2)

            label1 = label[:, :, 0:1]
            label2 = label[:, :, 1:2]
            label_change = label[:, :, 2:3]
            label = np.concatenate((label2, label1, label_change), axis=2)
        return [image, label]


class BCDNormalizeTransform:
    def __init__(self, mean, std):
        self.mean_array = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std_array = np.array(std, dtype=np.float32).reshape(1, 1, -1)
        
    def __call__(self, image, label):
        image = image.astype(np.float32) / 255.0
        label = np.ceil(label / 255.0).astype(np.float32)
        image = (image - self.mean_array) / self.std_array
        return [image, label]


class SCDBDANormalizeTransform:
    def __init__(self, mean, std):
        self.mean_array = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std_array = np.array(std, dtype=np.float32).reshape(1, 1, -1)
        
    def __call__(self, image, label):
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean_array) / self.std_array
        return [image, label]


class BCDToTensorTransform:
    def __init__(self, scale=1):
        self.scale = scale
        
    def __call__(self, image, label):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (w, h))
            label = cv2.resize(label, (int(w/self.scale), int(h/self.scale)), 
                              interpolation=cv2.INTER_NEAREST)
            
        image = image.transpose((2, 0, 1))            
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.LongTensor(np.array(label, dtype=np.int8)).unsqueeze(dim=0)
        
        return [image_tensor, label_tensor]


class SCDBDAToTensorTransform:
    def __init__(self, scale=1):
        self.scale = scale
        
    def __call__(self, image, label):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (w, h))
            label = cv2.resize(label, (int(w/self.scale), int(h/self.scale)),
                              interpolation=cv2.INTER_NEAREST)

        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image)
        label = label.transpose((2, 0, 1))
        label_tensor = torch.from_numpy(label)
        
        return [image_tensor, label_tensor]


class BCDTransforms:
    """
    Compact transformation utilities for remote sensing change detection datasets.
    
    This class provides factory methods for common image transformations used in
    remote sensing tasks, particularly for bi-temporal change detection.
    """
    
    # Default normalization parameters for remote sensing imagery
    DEFAULT_MEAN = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    DEFAULT_STD = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # ImageNet normalization parameters (for reference)
    IMAGENET_MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    IMAGENET_STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    
    @staticmethod
    def scale(width: int, height: int) -> Callable:
        """Fixed-size resize transform."""
        return ScaleTransform(width, height)
    
    @staticmethod
    def resize(min_size: Union[int, Sequence[int]], max_size: Optional[int] = None, 
               strict: bool = False) -> Callable:
        """Aspect-ratio preserving resize transform."""
        return ResizeTransform(min_size, max_size, strict)
    
    @staticmethod
    def random_crop_resize(crop_area: int) -> Callable:
        """Random crop and resize transform with 50% probability."""
        return RandomCropResizeTransform(crop_area)
    
    @staticmethod
    def random_flip() -> Callable:
        """Random horizontal and vertical flipping with 50% probability each."""
        return RandomFlipTransform()

    @staticmethod
    def random_exchange() -> Callable:
        """Randomly swap pre/post temporal images with 50% probability."""
        return BCDRandomExchangeTransform()
    
    @staticmethod
    def normalize(mean: Sequence[float], std: Sequence[float]) -> Callable:
        """Normalize image using mean and standard deviation."""
        return BCDNormalizeTransform(mean, std)
    
    @staticmethod
    def to_tensor(scale: int = 1) -> Callable:
        """Convert arrays to PyTorch tensors."""
        return BCDToTensorTransform(scale)

    @classmethod
    def compose(cls, transforms):
        """将多个转换函数组合成一个管道"""
        return ComposedTransforms(transforms)
    
    @classmethod
    def get_transform_pipelines(cls, args: Dict[str, Any]) -> Tuple[Callable, Callable]:
        """
        Create standard training and validation transform pipelines for remote sensing change detection.
        
        Args:
            args: Dictionary or object with in_width and in_height attributes
                - Must have in_width and in_height as int attributes
                - Can optionally have normalize_mean and normalize_std attributes
                
        Returns:
            Tuple of (train_transform, val_transform) functions
        """
        # Get normalization parameters, use defaults if not specified
        if hasattr(args, 'normalize_mean') and hasattr(args, 'normalize_std'):
            mean = args.normalize_mean
            std = args.normalize_std
        else:
            mean = cls.DEFAULT_MEAN
            std = cls.DEFAULT_STD
        
        # Calculate crop area based on input width
        crop_area = int(7.0 / 224.0 * args.in_width)
        
        # Training transforms with data augmentation
        train_transform = cls.compose([
            cls.normalize(mean=mean, std=std),
            cls.scale(width=args.in_width, height=args.in_height),
            cls.random_crop_resize(crop_area=crop_area),
            cls.random_flip(),
            cls.random_exchange(),
            cls.to_tensor()
        ])

        # Validation transforms without augmentation
        val_transform = cls.compose([
            cls.normalize(mean=mean, std=std),
            cls.scale(width=args.in_width, height=args.in_height),
            cls.to_tensor()
        ])
        
        return train_transform, val_transform


class SCDTransforms:
    """
    Compact transformation utilities for remote sensing change detection datasets.
    
    This class provides factory methods for common image transformations used in
    remote sensing tasks, particularly for bi-temporal change detection.
    """
    
    # Default normalization parameters for remote sensing imagery
    DEFAULT_MEAN = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    DEFAULT_STD = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # ImageNet normalization parameters (for reference)
    IMAGENET_MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    IMAGENET_STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    @staticmethod
    def scale(width: int, height: int) -> Callable:
        """Fixed-size resize transform."""
        return ScaleTransform(width, height)
    
    @staticmethod
    def resize(min_size: Union[int, Sequence[int]], max_size: Optional[int] = None, 
               strict: bool = False) -> Callable:
        """Aspect-ratio preserving resize transform."""
        return ResizeTransform(min_size, max_size, strict)
    
    @staticmethod
    def random_crop_resize(crop_area: int) -> Callable:
        """Random crop and resize transform with 50% probability."""
        return RandomCropResizeTransform(crop_area)
    
    @staticmethod
    def random_flip() -> Callable:
        """Random horizontal and vertical flipping with 50% probability each."""
        return RandomFlipTransform()

    @staticmethod
    def random_exchange() -> Callable:
        """Randomly swap pre/post temporal images with 50% probability."""
        return SCDRandomExchangeTransform()
    
    @staticmethod
    def normalize(mean: Sequence[float], std: Sequence[float]) -> Callable:
        """Normalize image using mean and standard deviation."""
        return SCDBDANormalizeTransform(mean, std)
    
    @staticmethod
    def to_tensor(scale: int = 1) -> Callable:
        """Convert arrays to PyTorch tensors."""
        return SCDBDAToTensorTransform(scale)

    @classmethod
    def compose(cls, transforms):
        """将多个转换函数组合成一个管道"""
        return ComposedTransforms(transforms)
    
    @classmethod
    def get_transform_pipelines(cls, args: Dict[str, Any]) -> Tuple[Callable, Callable]:
        """
        Create standard training and validation transform pipelines for remote sensing change detection.
        
        Args:
            args: Dictionary or object with in_width and in_height attributes
                - Must have in_width and in_height as int attributes
                - Can optionally have normalize_mean and normalize_std attributes
                
        Returns:
            Tuple of (train_transform, val_transform) functions
        """
        # Get normalization parameters, use defaults if not specified
        if hasattr(args, 'normalize_mean') and hasattr(args, 'normalize_std'):
            mean = args.normalize_mean
            std = args.normalize_std
        else:
            mean = cls.DEFAULT_MEAN
            std = cls.DEFAULT_STD
        
        # Calculate crop area based on input width
        crop_area = int(7.0 / 224.0 * args.in_width)
        
        # Training transforms with data augmentation
        train_transform = cls.compose([
            cls.normalize(mean=mean, std=std),
            cls.scale(width=args.in_width, height=args.in_height),
            cls.random_crop_resize(crop_area=crop_area),
            cls.random_flip(),
            cls.random_exchange(),
            cls.to_tensor()
        ])

        # Validation transforms without augmentation
        val_transform = cls.compose([
            cls.normalize(mean=mean, std=std),
            cls.scale(width=args.in_width, height=args.in_height),
            cls.to_tensor()
        ])
        
        return train_transform, val_transform
    

class BDATransforms:
    """
    Compact transformation utilities for remote sensing change detection datasets.
    
    This class provides factory methods for common image transformations used in
    remote sensing tasks, particularly for bi-temporal change detection.
    """
    
    # Default normalization parameters for remote sensing imagery
    DEFAULT_MEAN = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    DEFAULT_STD = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # ImageNet normalization parameters (for reference)
    IMAGENET_MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    IMAGENET_STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    
    @staticmethod
    def scale(width: int, height: int) -> Callable:
        """Fixed-size resize transform."""
        return ScaleTransform(width, height)
    
    @staticmethod
    def resize(min_size: Union[int, Sequence[int]], max_size: Optional[int] = None, 
               strict: bool = False) -> Callable:
        """Aspect-ratio preserving resize transform."""
        return ResizeTransform(min_size, max_size, strict)

    @staticmethod
    def random_crop_resize(crop_area: int) -> Callable:
        """Random crop and resize transform with 50% probability."""
        return RandomCropResizeTransform(crop_area)

    @staticmethod
    def random_flip() -> Callable:
        """Random horizontal and vertical flipping with 50% probability each."""
        return RandomFlipTransform()

    @staticmethod
    def random_exchange() -> Callable:
        """Randomly swap pre/post temporal images with 50% probability."""
        return SCDRandomExchangeTransform()

    @staticmethod
    def normalize(mean: Sequence[float], std: Sequence[float]) -> Callable:
        """Normalize image using mean and standard deviation."""
        return SCDBDANormalizeTransform(mean, std)

    @staticmethod
    def to_tensor(scale: int = 1) -> Callable:
        """Convert arrays to PyTorch tensors."""
        return SCDBDAToTensorTransform(scale)

    @classmethod
    def compose(cls, transforms):
        """将多个转换函数组合成一个管道"""
        return ComposedTransforms(transforms)
    
    @classmethod
    def get_transform_pipelines(cls, args: Dict[str, Any]) -> Tuple[Callable, Callable]:
        """
        Create standard training and validation transform pipelines for remote sensing change detection.
        
        Args:
            args: Dictionary or object with in_width and in_height attributes
                - Must have in_width and in_height as int attributes
                - Can optionally have normalize_mean and normalize_std attributes
                
        Returns:
            Tuple of (train_transform, val_transform) functions
        """
        # Get normalization parameters, use defaults if not specified
        if hasattr(args, 'normalize_mean') and hasattr(args, 'normalize_std'):
            mean = args.normalize_mean
            std = args.normalize_std
        else:
            mean = cls.DEFAULT_MEAN
            std = cls.DEFAULT_STD
        
        # Calculate crop area based on input width
        crop_area = int(7.0 / 224.0 * args.in_width)
        
        # Training transforms with data augmentation
        train_transform = cls.compose([
            cls.normalize(mean=mean, std=std),
            cls.scale(width=args.in_width, height=args.in_height),
            cls.random_crop_resize(crop_area=crop_area),
            cls.random_flip(),
            cls.random_exchange(),
            cls.to_tensor()
        ])

        # Validation transforms without augmentation
        val_transform = cls.compose([
            cls.normalize(mean=mean, std=std),
            cls.scale(width=args.in_width, height=args.in_height),
            cls.to_tensor()
        ])
        
        return train_transform, val_transform

def get_train_transform():
    """获取训练数据增强转换"""
    return A.Compose([
        A.RandomBrightnessContrast(p=0.3),       # 光照变化
        A.RandomGamma(p=0.3),                    # 伽马扰动模拟传感器变化
        A.GaussianBlur(p=0.2),                   # 锐度扰动
        A.ElasticTransform(alpha=1, p=0.1),      # 局部形变，模拟位移
        A.Rotate(limit=10, p=0.2),               # 小角度旋转
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height=256, width=256),
        ToTensorV2()
    ])

def get_val_transform():
    """获取验证数据转换"""
    return A.Compose([
        A.Resize(height=256, width=256),
        ToTensorV2()
    ])