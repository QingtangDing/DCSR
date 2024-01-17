# Not All Patches Are Equal: Hierarchical Dataset Condensation for Single Image Super-Resolution

The pytorch implementation of ''Not All Patches Are Equal: Hierarchical Dataset Condensation for Single Image Super-Resolution'', SPL 2023.

## Motivation

###   Super-resolution network parameter matching
- Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and crop the images into patches.
- Select patches with different complexities and  patches with different texture categories in DIV2K to compose different data subsets
-  Train super-resolution network using DIV2K and different data subsets. Note modifying the dataset path.
- Calculate the parameter matching error using the following command.
- ```bash
  python parameter_matching.py
  ```

## Method

<p align="center"> <img src="Figs/method.jpeg" width="100%"> </p>

## Requirements

- Python 3.7
- PyTorch == 1.7.0
- numpy
- skimage
- imageio
- matplotlib
- cv2

## Dataset condensation

### 1. Prepare condensation dataset
1.1 Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

1.2 Crop the HR images into image patches in 'data_path/DIV2K/HR'

### 2. Texture complexity-based condensation strategy
2.1 Measure the complexity of image patches 
``` bash
python texture_complexity_measurement.py
```

2.2 Remove the low-complexity image patches 

``` bash
python remove_low_complexity.py
```


### 3. Texture diversity-based condensation strategy
3.1 Cluster the image patches
```bash
python cluster_patches.py
```
3.2 Remove the image patches with similar textures in each cluster
```bash
python sample_patches.py
```

## Performance evaluation

### 1. Train super-resolution network with condensed dataset
```bash
python main.py --model EDSR --scale 2 --save DCSR_X2 --patch_size 96 --batch_size 16
```

### 2. Test model performance
2.1 Prepare test data
Download [benchmark datasets](https://github.com/xinntao/BasicSR/blob/a19aac61b277f64be050cef7fe578a121d944a0e/docs/Datasets.md) (e.g., Set5, Set14 and other test sets) and prepare HR/LR images in `/benchmark` following the example of `benchmark/Set5`
2.1 Test the performance

```bash
python main.py --model EDSR --data_test Set5 --scale 2  --test_only
```

## Citation
```
@ARTICLE{10305246,
  author={Ding, Qingtang and Liang, Zhengyu and Wang, Longguang and Wang, Yingqian and Yang, Jungang},
  journal={IEEE Signal Processing Letters}, 
  title={Not All Patches Are Equal: Hierarchical Dataset Condensation for Single Image Super-Resolution}, 
  year={2023},
  volume={30},
  number={},
  pages={1752-1756},
  doi={10.1109/LSP.2023.3329754}}
```

## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [ClassSR](https://github.com/XPixelGroup/ClassSR). We thank the authors for sharing the codes.
