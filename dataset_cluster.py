import argparse
import os.path
import shutil

import numpy as np
from tqdm import tqdm
import torch
from importlib import import_module
from experiment_test.network import DeepTen
from experiment_test import utils
from sklearn.cluster import KMeans


class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='Deep Encoding')
        parser.add_argument('--dataset', type=str, default='DIV2K',
                            help='training dataset (default: imagenet)')
        parser.add_argument('--base-size', type=int, default=None,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=224,
                            help='crop image size')
        # model params
        parser.add_argument('--model_name', type=str, default='EDSR', help="model_name")
        parser.add_argument('--model_path', type=str, default='./model_best.pth.tar',
                            help='the path of pretrained model')
        parser.add_argument('--rectify', action='store_true',
                            default=False, help='rectify convolution')
        parser.add_argument('--rectify-avg', action='store_true',
                            default=False, help='rectify convolution')
        # training hyper params
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='batch size for training (default: 128)')
        parser.add_argument('--workers', type=int, default=1,
                            metavar='N', help='dataloader threads')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true',
                            default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--verify', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--export', type=str, default=None,
                            help='put the path to resuming file if needed')
        # dataset
        parser.add_argument('--dir_data', type=str, default=r'E:\QingtangDing\dataset',
                            help='dataset directory')
        parser.add_argument('--data_train', type=str, default='DIV2K',
                            help='train dataset name')
        parser.add_argument('--data_test', type=str, default='Set5',
                            help='test dataset name:DIV2K,Set5,Set14,B100,Urban100')
        parser.add_argument('--data_range', type=str, default='1-800/801-810',
                            help='train/test data range')
        parser.add_argument('--no_augment', action='store_true', default=True,
                            help='do not use data_ augmentation')
        parser.add_argument('--patch_size', type=int, default=96,
                            help='output patch size')
        parser.add_argument('--n_colors', type=int, default=3,
                            help='number of color channels to use')
        parser.add_argument('--rgb_range', type=float, default=1,
                            help='maximum value of RGB')
        parser.add_argument('--scale', type=str, default="2", help='super resolution scale')
        parser.add_argument('--ext', type=str, default='sep', help='dataset file extension')
        # training set
        parser.add_argument('--test_every', type=int, default=1000, help='do test per every N batches')
        parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
        parser.add_argument('--cpu', action='store_true',
                            help='use cpu only')
        parser.add_argument('--precision', type=str, default='single',
                            choices=('single', 'half'),
                            help='FP precision for test (single | half)')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.scale = list(map(lambda x: int(x), args.scale.split('+')))
        args.data_train = args.data_train.split("+")
        args.data_test = args.data_test.split("+")
        return args


def get_dataloaders(args):
    module_name = args.dataset if args.dataset.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
    m = import_module('data.' + module_name.lower())
    trainset = getattr(m, module_name)(args, name=args.dataset, train=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_workers)
    return trainloader


def main():
    # init the args
    args = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # init dataloader
    trainloader = get_dataloaders(args)
    model = DeepTen(47, backbone='resnet50')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_path)
    model_dict['fc.weight'] = pretrained_dict['head.6.weight']
    model_dict['fc.bias'] = pretrained_dict['head.6.bias']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and (v.shape == model_dict[k].shape)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # print(model)
    if args.cuda:
        model.cuda(1)
    model.eval()
    tbar = tqdm(trainloader, desc='\r')
    file_names = []
    data_feature = None
    data_feature_path = None
    if data_feature_path is None:
        print("feature extraction")
        for batch_idx, (lr, hr, file_name) in enumerate(tbar):
            if args.cuda:
                hr = utils.prepare(args, (hr,))[-1]
            with torch.no_grad():
                output = model(hr)
            if batch_idx == 0:
                data_feature = output
            else:
                data_feature = torch.cat((data_feature, output), dim=0)
            file_names += file_name
        torch.save(data_feature.cpu(), './data_feature_flickr2k_x2.pt')
        np.save('./cluster_file_names_flickr2k_x2.npy', file_names)
    else:
        data_feature = torch.load(data_feature_path)
        file_names = np.load('./cluster_file_names_flickr2k_x2.npy')
    n_clusters = 15
    if data_feature.is_cuda:
        data_feature = data_feature.cpu()
    # 聚类
    data_feature_np = data_feature.numpy()
    kmeans = KMeans(n_clusters=15, init='k-means++', n_init=10, max_iter=300)
    kmeans.fit(data_feature_np)
    np.save('cluster_labels_' + str(n_clusters), kmeans.labels_)
    data_cluster_dir = r'E:\QingtangDing\dataset\Flickr2K\data_x2_cluster_' + str(n_clusters)
    for i in range(n_clusters):
        if not os.path.exists(os.path.join(data_cluster_dir, str(i))):
            os.makedirs(os.path.join(data_cluster_dir, str(i)))
    for i, label in enumerate(kmeans.labels_):
        shutil.copy(os.path.join(r'E:\QingtangDing\dataset\Flickr2K\train_HR_50', file_names[i] + '.png'),
                    os.path.join(data_cluster_dir, str(label), file_names[i] + '.png'))


if __name__ == "__main__":
    main()
