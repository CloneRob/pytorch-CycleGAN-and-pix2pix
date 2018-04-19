import os
import pickle
from os.path import join, basename, splitext
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from data.transform import Compose, ToTensor, RandomHorizontalFlip, RandomCrop, RandomZoom, ColorJitter, CenterCrop, Resize, Normalize
from glob import glob

import numpy as np
from PIL import Image


def normal_values(dataset):
    """Calculates the mean and std pixel from a given dataset
        Args:
            dataset: Pytorch Dataset
        Returns:
            mean, std arrays containing 3 floats each
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    pixels_in_image = None
    pixel_sum = np.zeros(3)

    for item, _ in loader:
        if not isinstance(item, np.ndarray):
            item = item.numpy()

        if pixels_in_image is None:
            pixels_in_image = item.shape[-2] * item.shape[-1]

        pixel_sum += np.sum(item, axis=(0, 2, 3))

    number_of_pixels = len(loader.dataset) * pixels_in_image
    mean = pixel_sum / number_of_pixels
    std = np.zeros(3)
    for item, _ in loader:
        if not isinstance(item, np.ndarray):
            item = item.numpy()
        item = np.transpose(item, (0, 2, 3, 1))
        d = (item - mean)**2
        std += np.sum(d, axis=(0, 1, 2))

    std = np.sqrt(std / number_of_pixels)

    output_dict = {'mean': mean, 'std': std}
    return output_dict


class Folder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, target_expansion=None):
        super(Folder, self).__init__(root, transform, target_transform)
        assert callable(target_expansion)
        self.target_expansion = target_expansion
        self.n_classes = len(self.classes)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, target_expansion) where target is class_index of the target class.
        """
        img, target = super(Folder, self).__getitem__(index)
        target_expansion = self.target_expansion(target)
        return img, target, target_expansion

    def __len__(self):
        return len(self.imgs)


class VOCSegmentation(torch.utils.data.Dataset):
    def __init__(self, in_root, target_root, id_list):
        super(VOCSegmentation, self).__init__()
        self.in_root = in_root
        self.tar_root = target_root
        self.ids = id_list
        self.transforms = None

    def set_transforms(self, transforms):
        assert isinstance(transforms, Compose)
        self.transforms = transforms

    @staticmethod
    def open_image(path, mode='RGB'):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if mode is not None:
                img = img.convert(mode=mode)
            return img

    def __getitem__(self, index):
        input_path = join(self.in_root, self.ids[index] + '.jpg')
        target_path = join(self.tar_root, self.ids[index] + '.png')
        input_img = self.open_image(input_path, mode='RGB')
        target_img = self.open_image(target_path, mode='P')

        input_img, target_img = self.transforms.tuple_transforms(
            input_img, target_img)

        return input_img, target_img

    def __len__(self):
        return len(self.ids)

    def __add__(self, other):
        raise NotImplementedError


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, data_dict):
        super(Dataset, self).__init__()
        self.root = root
        assert isinstance(data_dict, dict)

        self.idx_to_class = {i: k for i, k in enumerate(data_dict.keys())}
        self.class_to_idx = {k: i for i, k in self.idx_to_class.items()}
        self.ids = [(v, self.class_to_idx[k]) for k, v in data_dict.items()]

        self.data_dict = data_dict
        self.transforms = None

    def set_transforms(self, transforms):
        assert isinstance(transforms, Compose)
        self.transforms = transforms

    @staticmethod
    def open_image(path, mode='RGB'):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if mode is not None:
                img = img.convert(mode=mode)
            return img

    def __getitem__(self, index):
        ident, target = self.ids[index]

        input_path = join(self.root, ident + '.jpg')
        input_img = self.open_image(input_path, mode='RGB')

        input_img = self.transforms(input_img)

        return input_img, target

    def __len__(self):
        return len(self.ids)

    def __add__(self, other):
        raise NotImplementedError


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, mask_root, ids=None, img_fmt='.jpg', mask_fmt='.png'):
        super().__init__()
        self.image_root = image_root
        self.mask_root = mask_root
        self.img_fmt = img_fmt
        self.mask_fmt = mask_fmt

        if ids is None:
            paths = glob(join(image_root, '*' + img_fmt))
            ids = [basename(splitext(p)[0]) for p in paths]

        self.ids = ids
        self.transforms = None

    def set_transforms(self, transforms):
        assert isinstance(transforms, Compose)
        self.transforms = transforms

    @staticmethod
    def open_image(path, mode='RGB'):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if mode is not None:
                img = img.convert(mode=mode)
            return img

    def __getitem__(self, index):
        input_path = join(self.image_root, self.ids[index] + self.img_fmt)
        mask_path = join(self.mask_root, self.ids[index] + self.mask_fmt)

        inputp = self.open_image(input_path, mode='RGB')
        maskp = self.open_image(mask_path, mode='P')

        input_img, mask_img = self.transforms.tuple_transforms(inputp, maskp)
        mask_img = mask_img - 1

        return {'A': input_img, 'B': mask_img,
                'A_paths': input_path, 'B_paths': mask_path}

    def __len__(self):
        return len(self.ids)


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, root_a, root_b, ids=None, fmt_a='.jpg', fmt_b='.jpg'):
        super().__init__()
        self.root_a = root_a
        self.root_b = root_b
        self.fmt_a = fmt_a
        self.fmt_b = fmt_b

        if ids is None:
            paths_a = glob(join(root_a, '*' + fmt_a))
            paths_b = glob(join(root_b, '*' + fmt_b))
            ids_a = [basename(splitext(p)[0]) for p in paths_a]
            len_a = len(ids_a)
            ids_b = [basename(splitext(p)[0]) for p in paths_b]
            len_b = len(ids_b)

            if len_a > len_b:
                ids_a = ids_a[:len_b]
            elif len_b > len_a:
                ids_b = ids_b[:len_a]
            ids = [(a, b) for a, b in zip(ids_a, ids_b)]

        self.ids = ids
        self.transforms = None

    def set_transforms(self, transforms):
        assert isinstance(transforms, Compose)
        self.transforms = transforms

    @staticmethod
    def open_image(path, mode='RGB'):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if mode is not None:
                img = img.convert(mode=mode)
            return img

    def __getitem__(self, index):
        img1_id, img2_id = self.ids[index]
        img1_path = join(self.root_a, img1_id + self.fmt_a)
        img2_path = join(self.root_b, img2_id + self.fmt_b)

        img1 = self.open_image(img1_path, mode='RGB')
        img2 = self.open_image(img2_path, mode='RGB')

        im1, im2 = self.transforms.tuple_transforms(img1, img2)

        return im1, im2

    def __len__(self):
        return len(self.ids)


class ADE20K:
    def __init__(self, args):
        root = args.data
        image_root = join(root, 'images/{}')
        mask_root = join(root, 'masks/{}')

        rs = np.random.RandomState(args.seed)
        self.nv = {'mean':[.485, .456, .406], 'std':[.229, .224, .225]}

        def __mapping(m):
            npmask = np.array(m, np.int32, copy=False)
            mask = torch.from_numpy(npmask).float().unsqueeze(0)
            return mask

        self.train_ds = SegmentationDataset(image_root.format('training'), mask_root.format('training'))
        self.train_ds.set_transforms(ADE20K.transforms(rs, None, args.loadSize, args.fineSize))
        nv_file = join(args.cpbase, 'ADE20K.pickle')
        if os.path.exists(nv_file):
            with open(nv_file, 'rb') as nv:
                self.nv = pickle.load(nv)
        else:
            with open(nv_file, 'wb') as nv:
                self.nv = normal_values(self.train_ds)
                pickle.dump(self.nv, nv)

        print('Normalixation Constants: {}, {}'.format(self.nv['mean'], self.nv['std']))
        self.train_ds.set_transforms(ADE20K.transforms(rs, self.nv, args.loadSize, args.fineSize))

        print(len(self.train_ds))
        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)

        self.val_ds = SegmentationDataset(image_root.format('validation'), mask_root.format('validation'))
        self.val_ds.set_transforms(ADE20K.transforms(rs, self.nv, args.loadSize, args.fineSize, mode='test'))
        print(len(self.val_ds))


        self.val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=max(args.batch_size // 2, 1), shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)

    @staticmethod
    def transforms(rs, nv, load_size, fine_size, mode='train'):
        if mode is 'train':
            tfs = [RandomHorizontalFlip(rs),
                      RandomZoom((0.9, 1.15), rs),
                      Resize(load_size),
                      ColorJitter(rs),
                      RandomCrop(fine_size, rs),
                      ToTensor(mapping=mapping)]

        elif mode is 'test':
            tfs = [Resize(fine_size), CenterCrop(fine_size), ToTensor(mapping=mapping)]
        else:
            return None

        if nv is not None:
            tfs.append(Normalize(nv['mean'], nv['std']))

        return Compose(tfs)

    def dataloader(self):
        return self.train_loader, self.val_loader
    
    def denorm(self, img):
        for t, m, s in zip(img, self.nv['mean'], self.nv['std']):
            t.mul_(s).add_(m)
        return img 

def mapping(m):
    npmask = np.array(m, np.int32, copy=False)
    mask = torch.from_numpy(npmask).float().unsqueeze(0)
    return mask

class Cityscapes:
    def __init__(self, args):
        root = args.data
        image_root = join(root, 'images/{}')
        mask_root = join(root, 'masks/{}')

        rs = np.random.RandomState(args.seed)

        def extract_imgpaths(search_path):
            dir_data = [i for i in os.walk(search_path)][1:]
            ids = []
            for root, _, files in dir_data:
                _files = ['_'.join(f.split('_')[:3]) for f in files]
                ids.extend([join(basename(root), f) for f in _files])
            return ids

        ids = extract_imgpaths(image_root.format('train'))
        self.train_ds = SegmentationDataset(image_root.format('train'), mask_root.format('train'), ids=ids, img_fmt='_leftImg8bit.png', mask_fmt='_gtFine_labelIds.png')
        self.train_ds.set_transforms(Cityscapes.transforms(rs, None, args.loadSize, args.fineSize))
        nv_file = join(args.cpbase, 'cityscapes.pickle')
        if os.path.exists(nv_file):
            with open(nv_file, 'rb') as nv:
                self.nv = pickle.load(nv)
        else:
            with open(nv_file, 'wb') as nv:
                self.nv = normal_values(self.train_ds)
                pickle.dump(self.nv, nv)
        # self.nv['mean'] = torch.Tensor([0.5, 0.5, 0.5])
        # self.nv['std'] = torch.Tensor([0.5, 0.5, 0.5])
        print('Normalixation Constants: {}, {}'.format(self.nv['mean'], self.nv['std']))
        self.train_ds.set_transforms(Cityscapes.transforms(rs, self.nv, args.loadSize, args.fineSize))

        # self.train_loader = torch.utils.data.DataLoader(
        #     self.train_ds, batch_size=args.batch_size, shuffle=True,
        #     num_workers=args.workers, pin_memory=True, drop_last=True)

        ids = extract_imgpaths(image_root.format('val'))
        self.val_ds = SegmentationDataset(image_root.format('val'), mask_root.format('val'), ids=ids, img_fmt='_leftImg8bit.png', mask_fmt='_gtFine_labelIds.png')
        self.val_ds.set_transforms(Cityscapes.transforms(rs, self.nv, args.loadSize, args.fineSize, mode='test'))
        print(len(self.val_ds))


        # self.val_loader = torch.utils.data.DataLoader(
        #     self.val_ds, batch_size=max(args.batch_size // 2, 1), shuffle=True,
        #     num_workers=args.workers, pin_memory=True, drop_last=True)

        if args.isTrain:
            return self.train_ds
        else:
            return self.val_ds

    def dataloader(self):
        return self.train_loader, self.val_loader
    
    def denorm(self, img):
        for t, m, s in zip(img, self.nv['mean'], self.nv['std']):
            t.mul_(s).add_(m)
        return img 

    @staticmethod
    def transforms(rs, nv, load_size, fine_size, mode='train'):
        if mode is 'train':
            tfs = [RandomHorizontalFlip(rs),
                      RandomZoom((0.9, 1.15), rs),
                      Resize(load_size),
                      ColorJitter(rs),
                      RandomCrop(fine_size, rs),
                      ToTensor(mapping=mapping)]

        elif mode is 'test':
            tfs = [Resize(fine_size), CenterCrop(fine_size), ToTensor(mapping=mapping)]
        else:
            return None

        if nv is not None:
            tfs.append(Normalize(nv['mean'], nv['std']))

        return Compose(tfs)
