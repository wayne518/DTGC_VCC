# DTGC

Official implementation of **"Dual-branch Temporal Interaction and Global Correlation Network for Video Crowd Counting" (DTGC)**.

## Dataset

- Bus: [BaiduNetDisk](https://pan.baidu.com/s/1FR7PMrdhpNB2OgkY_QbbDw?pwd=ir6n)  (pwd: ir6n)
- Canteen: [BaiduNetDisk](https://pan.baidu.com/s/18XtesjJTBolXMwHZFoazVw?pwd=yi7b)  (pwd: yi7b)
- Classroom: [BaiduNetDisk](https://pan.baidu.com/s/1ZbD3aLNuu7syw86a7UQe-g?pwd=z3q8)  (pwd: z3q8)

## Install dependencies

torch >= 1.0, torchvision, opencv, numpy, scipy, etc.

## Take training and testing of Bus dataset for example:

1. Download Bus.
2. Preprocess Bus to generate ground-truth density maps.

```shell
python generate_h5.py
```

3. Divide the last 10% of the training set into the validation set. The folder structure should look like this:

```shell
Bus/
├── train/
│   ├── ground_truth/
│   │   ├── xxx_0.h5
│   │   ├── xxx_10.h5
│   │   └── ...
│   └── images/
│       ├── xxx_0.jpg
│       ├── xxx_10.jpg
│       └── ...
├── val/
│   ├── ground_truth/
│   └── images/
├── test/
│   ├── ground_truth/
│   └── images/
└── bus_roi.npy
```

4. Train Bus.

```bash
python train.py --data-dir (dataset path)  --roi-path (roi path) 
```

5. Test Bus.

```bash
python test.py --data-dir (dataset path)  --roi-path (roi path)  --save-dir (weight's path)
```

