from typing import Dict
import unittest

import os
from glob import glob
import torch
from torch.utils.data import DataLoader

allowed = [
    'boxes',
    'boxes_class',
    'points',
    'colors',
    'normals',
    'pose',
    'semantic',
    'instance',
    'image',
    'extra',
    'filename',
]


class Base(unittest.TestCase):

    def __init__(self, methodName, path_to_folder=''):
        super().__init__(methodName)
        self.path_to_folder = path_to_folder
        self.folder_to_module = path_to_folder.replace('/', '.')
        self.only_testing = os.environ.get('dataloader')
        self.dataloaders = []

    def main_tester(self):
        self.dataset_discovering()
        self.dataset_looper()

    def dataset_tester(self, dataset):

        # checking arguments
        args_name = self.get_args_names(dataset)
        self.assertIn('path_in_dataserver', args_name, 'Each IPB_Base should have the argument path_in_dataserver')
        self.assertIn('data_source', args_name, 'Each IPB_Base should have the argument data_source')
        # self.assertIn('split', args_name, 'Each IPB_Base should have the argument split')
        self.assertIsNotNone(dataset.path_in_dataserver, 'There must be a path in the dataserver.')

        # checking __getitem__
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate)
        for item in loader:
            self.assertTrue(isinstance(item, Dict), 'Item is not a Dict.')
            for key in item.keys():
                self.assertIn(key, allowed)

            # per-sensor tests
            if self.get_package_name(dataset) == 'rgb':
                self.rgb_tester(item)
            elif self.get_package_name(dataset) == 'rgbd':
                self.rgbd_tester(item)
            elif self.get_package_name(dataset) == 'lidar':
                self.lidar_tester(item)
            elif self.get_package_name(dataset) == 'pointcloud':
                self.pointcloud_tester(item)
            else:
                self.assertTrue(False, 'package name not recognized')
            break

        print(type(dataset).__name__, 'success')

    def dataset_looper(self):
        for handler in self.dataloaders:
            # checks if we are testing a specific loader
            # TODO: put this check in dataset_discovering?
            args_name = self.get_args_names(handler)
            self.assertIn('unittesting', args_name, 'Each IPB_Base should have the argument unittesting, defaulting to False')
            if self.only_testing:
                class_name = handler.__name__
                if class_name == self.only_testing:
                    dataset = handler(unittesting=True)
                    self.dataset_tester(dataset)
            # test'em all
            else:
                dataset = handler()
                self.dataset_tester(unittesting=True)

    def dataset_discovering(self):

        importlib = __import__('importlib')
        for file in glob(os.path.join(os.path.abspath(self.path_to_folder), "*.py")):

            # avoid imports that raise error, don't know if this is a good idea
            # doing it to avoid importing stuff from file which we are not testing
            name = os.path.splitext(os.path.basename(file))[0]
            try:
                module = importlib.import_module(self.folder_to_module + name)
            except Exception as e:
                print('Skipping {} because of missing libraries: {}'.format(name, e))
                continue

            for member in dir(module):
                # checking that member is a class
                handler_class = getattr(module, member)
                if hasattr(handler_class, 'mro'):
                    # not testing IPB_Base itself
                    if handler_class.__name__ == 'IPB_Base':
                        continue
                    # checking that all other torch.Datasets are IPB_Base
                    super_classes = handler_class.mro()[1:]
                    if 'Dataset' in [h.__name__ for h in super_classes]:
                        self.assertIn('IPB_Base', [h.__name__ for h in super_classes], 'Found a torch.Dataset which is not an IPB_Base (grand-)child.')
                        # adding all IPB_Base to a list for later testing
                        if 'IPB_Base' in [h.__name__ for h in super_classes]:
                            self.dataloaders.append(handler_class)

    def image_tester(self, item, dim):
        image = item['image']
        image_shape = item['image'].shape
        self.assertIsInstance(image, torch.Tensor, "item['image'] should be of type torch.Tensor")
        self.assertIn(image.dtype, [torch.float, torch.double], "type item['image'] should be of type torch.float or torch.double")
        self.assertGreaterEqual(image.min().item(), 0, "item['image'] should be in range (0,1)")
        if dim == 3:
            self.assertLessEqual(image.max().item(), 1, "item['image'] should be in range (0,1)")
        elif dim == 4:
            rgb = image[:, :3, :, :]
            self.assertLessEqual(rgb.max().item(), 1, "item['image'] should be in range (0,1)")
            depth = image[:, 3, :, :]
            self.assertEqual(depth.isnan().sum(), 0, "nan found in depth map. Invalid depth should be =0")
        else:
            self.assertTrue(False, "we only support RGB (dim=3) or RGBD (dim=4) images, found dim={}".format(dim))

        if 'semantic' in item.keys():
            sem = item['semantic']
            sem_shape = item['semantic'].shape
            self.assertEqual(sem_shape[-2:], image_shape[-2:])
            self.label_tester(sem, 'semantic')

        if 'instance' in item.keys():
            ins = item['instance']
            ins_shape = item['instance'].shape
            self.assertEqual(ins_shape[-2:], image_shape[-2:])
            self.label_tester(ins, 'instance')

        if 'boxes' in item.keys():
            self.assertIn('boxes_class', item.keys())
            self.box_tester(item, dim=2)

        elif 'boxes_class' in item.keys():
            self.assertIn('boxes', item.keys())
            self.box_tester(item, dim=2)

    def box_tester(self, item, dim):
        boxes = item['boxes']
        boxes_class = item['boxes_class']
        self.assertIn(boxes.dtype, [torch.float, torch.double], "type item['boxes'] should be of type torch.float or torch.double")
        self.assertIn(boxes_class.dtype, [torch.short, torch.int, torch.long], "type item['boxes_class'] should be of torch.short, torch.int or torch.long")
        self.assertEqual(len(boxes_class[0]), len(boxes[0]), "item['boxes'] and item['boxes_classes'] should have the same length")
        self.assertEqual(boxes_class[0][0].ndim, 0, "item['boxes_class'] should only contains a single values per bounding box")
        if dim == 2:
            self.assertEqual(len(boxes[0][0]), 4, "item['boxes'] should only contains 4 values per bounding box")
            image_shape = item['image'].shape
            top_right_x = boxes[0][:, 0]
            self.assertGreaterEqual(top_right_x.min(), 0, "top right x coordinate should be greater or equal than 0")
            top_right_y = boxes[0][:, 1]
            self.assertGreaterEqual(top_right_y.min(), 0, "top right y coordinate should be greater or equal than 0")
            bottom_left_x = boxes[0][:, 0]
            self.assertLess(bottom_left_x.min(), image_shape[2], "bottom left x coordinate should be smaller than image width")
            bottom_left_y = boxes[0][:, 1]
            self.assertLess(bottom_left_y.min(), image_shape[3], "bottom left y coordinate should be smaller than image height")

        elif dim == 3:
            self.assertEqual(len(boxes[0][0]), 8, "item['boxes'] should only contains 8 values per bounding box")
            # TODO: check box coordinates at some point
        else:
            self.assertTrue(False, "we only support 2D or 3D bounding boxes, found {}".format(dim))

    def label_tester(self, label, key):
        self.assertIsInstance(label, torch.Tensor, "item['{}'] should be of type torch.Tensor".format(key))
        self.assertIn(label.dtype, [torch.short, torch.int, torch.long], "type item['{}'] should be of type torch.short, torch.int or torch.long".format(key))
        self.assertGreaterEqual(label.min().item(), 0, "item['{}'] should not contain negative values)".format(key))

    def rgb_tester(self, item):
        image = item['image']
        image_shape = image.shape
        if image.ndim == 3:
            self.assertEqual(image_shape[0], 3)
        elif image.ndim == 4:
            self.assertEqual(image_shape[1], 3)
        else:
            self.assertTrue(False, "Expected image shape (N, 3, H, W) or (3, H, W)")
        self.image_tester(item, dim=3)

    def rgbd_tester(self, item):
        image = item['image']
        image_shape = item['image'].shape
        if image.ndim == 3:
            self.assertEqual(image_shape[0], 4)
        elif image.ndim == 4:
            self.assertEqual(image_shape[1], 4)
        else:
            self.assertTrue(False, "Expected image shape (N, 4, H, W) or (4, H, W)")
        self.image_tester(item, dim=4)

    def lidar_tester(self, item):
        self.pointcloud_tester(item)

    def pointcloud_tester(self, item):
        points = item['points'].squeeze()
        self.assertEqual(len(points[0]), 3, "We only support 3-dimensional point clouds")
        self.assertIsInstance(points, torch.Tensor, "item['points'] should be of type torch.Tensor")
        self.assertIn(points.dtype, [torch.float, torch.double], "type item['points'] should be of type torch.float or torch.double")

        if 'colors' in item.keys():
            colors = item['colors'].squeeze()
            self.assertIsInstance(colors, torch.Tensor, "item['colors'] should be of type torch.Tensor")
            self.assertEqual(len(colors), len(points), "item['points'] and item['colors'] should have the same length")
            self.assertEqual(len(colors[0]), 3, "item['colors'] should be a 3-dimensional tensor")
            self.assertIn(colors.dtype, [torch.float, torch.double], "type item['points'] should be of type torch.float or torch.double")
            self.assertGreaterEqual(colors.min().item(), 0, "item['colors'] should be in range (0,1)")
            self.assertLessEqual(colors.max().item(), 1, "item['colors'] should be in range (0,1)")

        if 'normals' in item.keys():
            normals = item['normals'].squeeze()
            self.assertIsInstance(normals, torch.Tensor, "item['normals'] should be of type torch.Tensor")
            self.assertEqual(len(normals), len(points), "item['points'] and item['normals'] should have the same length")
            self.assertEqual(len(normals[0]), 3, "item['normals'] should be a 3-dimensional tensor")
            self.assertIn(normals.dtype, [torch.float, torch.double], "type item['points'] should be of type torch.float or torch.double")

        if 'semantic' in item.keys():
            sem = item['semantic'].squeeze()
            self.assertEqual(len(sem), len(points), "item['points'] and item['semantic'] should have the same length")
            self.label_tester(sem, 'semantic')

        if 'instance' in item.keys():
            sem = item['instance'].squeeze()
            self.assertEqual(len(sem), len(points), "item['points'] and item['instance'] should have the same length")
            self.label_tester(sem, 'instance')

        if 'boxes' in item.keys():
            self.assertIn('boxes_class', item.keys())
            self.box_tester(item, dim=3)

        elif 'boxes_class' in item.keys():
            self.assertIn('boxes', item.keys())
            self.box_tester(item, dim=3)

    @staticmethod
    def get_package_name(dataset):
        module_str = dataset.__module__
        pkg_name = module_str.split('.')[-2]
        return pkg_name

    @staticmethod
    def get_args_names(dataset):
        args_names = dataset.__init__.__code__.co_varnames[:dataset.__init__.__code__.co_argcount]
        return args_names