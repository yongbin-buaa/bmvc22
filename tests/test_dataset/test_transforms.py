import unittest.mock as mock

import numpy as np
import torchvision.transforms as TF
from PIL import Image

import mmocr.datasets.pipelines.transforms as transforms
from mmdet.core import BitmapMasks


@mock.patch('%s.transforms.np.random.random_sample' % __name__)
@mock.patch('%s.transforms.np.random.randint' % __name__)
def test_random_crop_instances(mock_randint, mock_sample):

    img_gt = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 1],
                       [0, 0, 1, 1, 1], [0, 0, 1, 1, 1]])
    # test target is bigger than img size in sample_offset
    mock_sample.side_effect = [1]
    rci = transforms.RandomCropInstances(6, instance_key='gt_kernels')
    (i, j) = rci.sample_offset(img_gt, (5, 5))
    assert i == 0
    assert j == 0

    # test the second branch in sample_offset

    rci = transforms.RandomCropInstances(3, instance_key='gt_kernels')
    mock_sample.side_effect = [1]
    mock_randint.side_effect = [1, 2]
    (i, j) = rci.sample_offset(img_gt, (5, 5))
    assert i == 1
    assert j == 2

    mock_sample.side_effect = [1]
    mock_randint.side_effect = [1, 2]
    rci = transforms.RandomCropInstances(5, instance_key='gt_kernels')
    (i, j) = rci.sample_offset(img_gt, (5, 5))
    assert i == 0
    assert j == 0

    # test the first bracnh is sample_offset

    rci = transforms.RandomCropInstances(3, instance_key='gt_kernels')
    mock_sample.side_effect = [0.1]
    mock_randint.side_effect = [1, 1]
    (i, j) = rci.sample_offset(img_gt, (5, 5))
    assert i == 1
    assert j == 1

    # test crop_img(img, offset, target_size)

    img = img_gt
    offset = [0, 0]
    target = [6, 6]
    crop = rci.crop_img(img, offset, target)
    assert np.allclose(img, crop[0])
    assert np.allclose(crop[1], [0, 0, 5, 5])

    target = [3, 2]
    crop = rci.crop_img(img, offset, target)
    assert np.allclose(np.array([[0, 0], [0, 0], [0, 0]]), crop[0])
    assert np.allclose(crop[1], [0, 0, 2, 3])

    # test __call__
    rci = transforms.RandomCropInstances(3, instance_key='gt_kernels')
    results = {}
    gt_kernels = [img_gt, img_gt.copy()]
    results['gt_kernels'] = BitmapMasks(gt_kernels, 5, 5)
    results['img'] = img_gt.copy()
    results['mask_fields'] = ['gt_kernels']
    mock_sample.side_effect = [0.1]
    mock_randint.side_effect = [1, 1]
    output = rci(results)
    print(output['img'])
    target = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]])
    assert output['img_shape'] == (3, 3)

    assert np.allclose(output['img'], target)

    assert np.allclose(output['gt_kernels'].masks[0], target)
    assert np.allclose(output['gt_kernels'].masks[1], target)


@mock.patch('%s.transforms.np.random.random_sample' % __name__)
def test_scale_aspect_jitter(mock_random):
    img_scale = [(3000, 1000)]  # unused
    ratio_range = (0.5, 1.5)
    aspect_ratio_range = (1, 1)
    multiscale_mode = 'value'
    long_size_bound = 2000
    short_size_bound = 640
    resize_type = 'long_short_bound'
    keep_ratio = False
    jitter = transforms.ScaleAspectJitter(
        img_scale=img_scale,
        ratio_range=ratio_range,
        aspect_ratio_range=aspect_ratio_range,
        multiscale_mode=multiscale_mode,
        long_size_bound=long_size_bound,
        short_size_bound=short_size_bound,
        resize_type=resize_type,
        keep_ratio=keep_ratio)
    mock_random.side_effect = [0.5]

    # test sample_from_range

    result = jitter.sample_from_range([100, 200])
    assert result == 150

    # test _random_scale
    results = {}
    results['img'] = np.zeros((4000, 1000))
    mock_random.side_effect = [0.5, 1]
    jitter._random_scale(results)
    # scale1 0.5， scale2=1 scale =0.5  650/1000, w, h
    # print(results['scale'])
    assert results['scale'] == (650, 2600)


@mock.patch('%s.transforms.np.random.random_sample' % __name__)
def test_random_rotate(mock_random):

    mock_random.side_effect = [0.5, 0]
    results = {}
    img = np.random.rand(5, 5)
    results['img'] = img.copy()
    results['mask_fields'] = ['masks']
    gt_kernels = [results['img'].copy()]
    results['masks'] = BitmapMasks(gt_kernels, 5, 5)

    rotater = transforms.RandomRotateTextDet()

    results = rotater(results)
    assert np.allclose(results['img'], img)
    assert np.allclose(results['masks'].masks, img)


def test_color_jitter():
    img = np.ones((64, 256, 3), dtype=np.uint8)
    results = {'img': img}

    pt_official_color_jitter = TF.ColorJitter()
    output1 = pt_official_color_jitter(img)

    color_jitter = transforms.ColorJitter()
    output2 = color_jitter(results)

    assert np.allclose(output1, output2['img'])


def test_affine_jitter():
    img = np.ones((64, 256, 3), dtype=np.uint8)
    results = {'img': img}

    pt_official_affine_jitter = TF.RandomAffine(degrees=0)
    output1 = pt_official_affine_jitter(Image.fromarray(img))

    affine_jitter = transforms.AffineJitter(
        degrees=0,
        translate=None,
        scale=None,
        shear=None,
        resample=False,
        fillcolor=0)
    output2 = affine_jitter(results)

    assert np.allclose(np.array(output1), output2['img'])
