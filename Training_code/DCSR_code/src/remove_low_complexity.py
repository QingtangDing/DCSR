"""A multi-thread tool to crop large images to sub-images for faster IO."""
import os
import os.path as osp
import sys
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils.util import ProgressBar  # noqa: E402
import data.util as data_util  # noqa: E402
import pickle


# Then we densely crop 1.59M sub-images with size 32 × 32 from LR images.


def main():
    file_name = "./DIV2K_HR.pkl"
    f = open(file_name, "rb+")
    with open(file_name, "rb+") as f:
        data = pickle.load(f)
    data.sort()
    num_patch = len(data)
    sample_num_patches = int(0.5 * num_patch)
    gradient_theshold = data[sample_num_patches - 1]

    mode = 'pair'  # single (one input folder) | pair (extract corresponding GT and LR pairs)
    opt = {}
    opt['n_thread'] = 5
    opt['compression_level'] = 3  # 3 is the default value in cv2
    if mode == 'pair':
        # cut training data
        GT_folder = r'E:\QingtangDing\dataset\DIV2K\DIV2K_train_HR'  # fix to your path
        LR_folder = r'E:\QingtangDing\dataset\DIV2K\DIV2K_LR_bicubic\X2'  # fix to your path
        save_GT_folder = r'E:\QingtangDing\dataset\DIV2K\train_HR_50'
        save_LR_folder = r'E:\QingtangDing\dataset\DIV2K\train_LR_full\X2'

        scale_ratio = 2
        crop_sz = 96  # the size of each sub-image (GT)
        step = 96  # step of the sliding crop window (GT)
        thres_sz = 0  # size threshold
        ########################################################################
        # check that all the GT and LR images have correct scale ratio
        img_GT_list = data_util._get_paths_from_images(GT_folder)
        img_LR_list = data_util._get_paths_from_images(LR_folder)
        assert len(img_GT_list) == len(img_LR_list), 'different length of GT_folder and LR_folder.'
        for path_GT, path_LR in zip(img_GT_list, img_LR_list):
            img_GT = Image.open(path_GT)
            img_LR = Image.open(path_LR)
            w_GT, h_GT = img_GT.size
            w_LR, h_LR = img_LR.size
            assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR width [{:d}] for {:s}.'.format(
                # noqa: E501
                w_GT, scale_ratio, w_LR, path_GT)
            assert h_GT / h_LR == scale_ratio, 'GT height [{:d}] is not {:d}X as LR height [{:d}] for {:s}.'.format(
                # noqa: E501
                h_GT, scale_ratio, h_LR, path_GT)
        # check crop size, step and threshold size
        assert crop_sz % scale_ratio == 0, 'crop size is not {:d}X multiplication.'.format(
            scale_ratio)
        assert step % scale_ratio == 0, 'step is not {:d}X multiplication.'.format(scale_ratio)
        assert thres_sz % scale_ratio == 0, 'thres_sz is not {:d}X multiplication.'.format(
            scale_ratio)
        print('process GT...')
        opt['input_folder'] = GT_folder
        opt['image_type'] = 'HR'
        opt['save_folder'] = save_GT_folder
        opt['crop_sz'] = crop_sz
        opt['step'] = step
        opt['thres_sz'] = thres_sz
        opt["gradient_threshold"] = gradient_theshold
        extract_signle(opt)
        # print('process LR...')
        # opt['input_folder'] = LR_folder
        # opt['image_type'] = 'LR'
        # opt['save_folder'] = save_LR_folder
        # opt['crop_sz'] = crop_sz // scale_ratio
        # opt['step'] = step // scale_ratio
        # opt['thres_sz'] = thres_sz // scale_ratio
        # opt["gradient_threshold"] = gradient_theshold_LR
        # extract_signle(opt)
        assert len(data_util._get_paths_from_images(save_GT_folder)) == len(
            data_util._get_paths_from_images(
                save_LR_folder)), 'different length of save_GT_folder and save_LR_folder.'
    else:
        raise ValueError('Wrong mode.')


def extract_signle(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    grad_thres = opt["gradient_threshold"]
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        # sys.exit(1)
    img_list = data_util._get_paths_from_images(input_folder)
    # img_list = img_list[0:800]

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(opt['n_thread'])

    for path in img_list:
        pool.apply_async(worker, args=(path, opt, grad_thres), callback=update).get()
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, opt, grad_thres):
    crop_sz = opt['crop_sz']
    step = opt['step']
    thres_sz = opt['thres_sz']
    img_name = osp.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    img_grad = []
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
            grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
            grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
            grad_x = abs(grad_x)
            grad_y = abs(grad_y)
            grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
            grad_mean = np.mean(grad)
            img_grad.append(grad_mean)
            if grad_mean > grad_thres:
                cv2.imwrite(
                    osp.join(opt['save_folder'],
                             img_name.replace('.png', '_s{:03d}.png'.format(index))), crop_img,
                    [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
            else:
                pass
            # cv2.imwrite(osp.join(opt['save_folder'], img_name.replace('.png', '_s{:03d}.png'.format(index))), crop_img,
            #             [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    print('Processing {:s} ...'.format(img_name))
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
