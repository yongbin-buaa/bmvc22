from mmdet.datasets.builder import PIPELINES
from torchvision.transforms import ColorJitter, RandomAffine
from torchvision.transforms import RandomPerspective, \
    RandomAffine, RandomRotation, GaussianBlur, ColorJitter
from mmdet.datasets.pipelines.loading import LoadImageFromFile
from PIL import Image
import numpy as np
import random
import cv2
import mmcv
import os.path as osp


@PIPELINES.register_module()
class LoadRGBImageFromFile(LoadImageFromFile):
    '''The original `LoadImageFromFile` class read image with BGR channels,
    it is not convenient to apply image transforms.
    '''
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        super().__init__(
            to_float32=to_float32,
            color_type=color_type,
            file_client_args=file_client_args
        )

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type, channel_order='rgb')
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class CustomIdentityTransform:
    def __init__(self):
        pass

    def __call__(self, results):
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class CustomRandomRotatation:
    def rotate_img(self, img, angle):
        ''' img is numpy array
        '''
        h, w = img.shape[:2]
        (center_x, center_y) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w / 2) - center_x
        M[1, 2] += (new_h / 2) - center_y
        return cv2.warpAffine(img, M, (new_w, new_h))

    def __init__(self, degrees=5, min_degrees=None, max_degrees=None):
        assert ((degrees is None) and isinstance(min_degrees, int) and isinstance(max_degrees, int))\
            or isinstance(degrees, int)
        if min_degrees is None:
            self.min_degrees = -degrees
        if max_degrees is None:
            self.max_degrees = degrees

    def __call__(self, results):
        img = results['img']
        angle = random.randint(self.min_degrees, self.max_degrees)
        img = self.rotate_img(img, angle)
        results['img'] = img
        results['img_shape'] = img.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class CustomPILTransforms:
    transform_names = dict(
        ColorJitter=ColorJitter,
        RandomPerspective=RandomPerspective,
        RandomAffine=RandomAffine,
        RandomRotation=CustomRandomRotatation,
        GaussianBlur=GaussianBlur,
        Identity=CustomIdentityTransform
    )
    def __init__(self, name="None", **kwargs):
        assert name in CustomPILTransforms.transform_names
        self.name = name
        self.transform = CustomPILTransforms.transform_names[name](**kwargs)

    def __call__(self, results):
        if self.name == "RandomRotation":
            return self.transform(results)
        else:
            img = Image.fromarray(results['img'])
            img = self.transform(img)
            results['img'] = np.asarray(img)
            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class CustomRandomApplyTransforms:
    def __init__(self, transforms=[], **kwargs):
        '''
        Example args:
            transforms = [
                dict(name="ColorJitter", brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                dict(name="RandomAffine", degrees=0.5, translate=(.1, .1), scale=(.95, 1.0)),
                dict(name="RandomRotation", degrees=5),
                dict(name="RandomPerspective"),
                dict(name="GaussianBlur", kernel_size=5)
            ]
        '''
        assert len(transforms) > 0
        assert all([isinstance(i, dict) for i in transforms])
        self.transforms = []
        for d in transforms:
            self.transforms.append(CustomPILTransforms(**d))

    def __call__(self, results):
        transform_func = random.choice(self.transforms)
        results = transform_func(results)
        results['data_aug'] = str(transform_func.name)
        return results


if __name__ == '__main__':
    from PIL import Image
    loader = LoadRGBImageFromFile()

    transforms = [
        dict(name="ColorJitter", brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        dict(name="RandomAffine", degrees=0.5, translate=(.1, .1), scale=(.95, 1.0)),
        dict(name="RandomRotation", degrees=5),
        dict(name="RandomPerspective"),
        dict(name="GaussianBlur", kernel_size=5)
    ]
    transform = CustomRandomApplyTransforms(transforms=transforms)

    results = dict(
        img_info=dict(
            filename='36.png',
            text='03/09/2009'
        ),
        img_prefix='/home/yongbin/gpfs/models/mmocr/data/mixture/ART/val',
        text='03/09/2009'
    )
    results = loader(results)
    print(results['img'].shape)
    results = transform(results)
    print(results['img'].shape)
