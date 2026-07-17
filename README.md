# ReCoFuse: Ultra-robust Image Fusion via Restorative Multi-modal Diffusion Reciprocal Coupling
This repository is the official implementation of the CVPR 2026 paper: [ReCoFuse: Ultra-Robust Image Fusion via Restorative Multi-Modal Diffusion Reciprocal Coupling](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_ReCoFuse_Ultra-Robust_Image_Fusion_via_Restorative_Multi-Modal_Diffusion_Reciprocal_Coupling_CVPR_2026_paper.pdf)

## ⚙️ Environmental Installation

```
conda create -n ReCoFuse python=3.9 -y
conda activate ReCoFuse
```

It is recommended to use PyTorch ≥ 1.13.0.
Please install a PyTorch version that matches your CUDA setup.
You may refer to the official PyTorch website or use the installation commands:

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Python packages:

```
pip install -r requirements.txt
```

---

## ❄️ Test

### Prepare Dataset

Please place the data in the following path:

```
./datasets/
```

After placing the data, please modify the file:

```
./option/test/ReCoFuse_test.yaml
```

Specifically, update the paths for

```
dataroot_GT_X, dataroot_LQ_X, dataroot_GT_Y, and dataroot_LQ_Y,
```

which correspond to the high-quality (GT) and low-quality (LQ) images used by different modules.

If you do not have GT images, you can set only the low-quality paths and leave the GT paths empty, for example:

```
dataroot_GT_X: ~
dataroot_GT_Y: ~
```

### pre-trained weights

We provide pre-trained model parameters. You can download them from [Google Drive](https://drive.google.com/drive/folders/1LAQ-klAMdqvrf1u1AYO-whoOWVmQ33am?usp=drive_link), then follow the instructions in:

```
./pretrained_models/
```

### Run

You can modify the parameter settings in ReCoFuse_test.yaml. Then run the code.

```
python test.py -opt=options/test/ReCoFuse_test.yml
```

## 🔥 Train

### Prepare Dataset

Please place the data in the following path:

```
./datasets/
```

Training requires both high-quality and low-quality  images for different modalities. Make sure that all modalities have the corresponding GT and LQ data properly organized.

For training the AutoEncoder and the two single-branch networks, you can refer to the training pipelines of [**OmniFuse**](https://github.com/HaoZhang1018/OmniFuse) and [**IRSDE**](https://github.com/Algolzw/image-restoration-sde) as examples.

### train Fusion model

Place the pretrained weights for the AutoEncoder and the two single-branch networks in:

```
./pretrained_models/
```

Then, configure the following fields in:

```
./option/test/ReCoFuse_train.yaml
```

* `pretrain_model_L`
* `pretrain_model_VIS`
* `pretrain_model_IR`

After setting the pretrained models, start training with:

```bash
python train.py -opt=options/train/ReCoFuse_train.yml
torchrun --nproc_per_node=2 train.py -opt=options/train/ReCoFuse_train.yml --launcher pytorch
```
