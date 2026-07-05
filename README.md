# Melanoma Segmentation — U-Net

Skin lesion segmentation using U-Net with EfficientNet-B0 encoder,
trained on ISIC 2018 Task 1 dataset.

## Results
| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1     | 0.2058     | 0.1386   |
| 2     | 0.1346     | 0.1583   |
| 3     | 0.1195     | 0.1126   |
| 4     | 0.1091     | 0.1340   |
| 5     | 0.0972     | 0.1194   |

## Stack
- PyTorch
- segmentation-models-pytorch
- EfficientNet-B0 encoder (pretrained ImageNet)
- DiceLoss
- Dataset: ISIC 2018 Task 1 (2000 train / 594 val)
- Trained on Kaggle GPU T4 x2

## Model
Hosted on Hugging Face: Ai-Adam-Six-Sigma/melanoma-segmentation

## Getting Started

### Option A: Kaggle (recommended — no GPU or disk space required)
1. Go to https://www.kaggle.com/notebooks and create New Notebook
2. Add dataset: `tschandl/isic2018-challenge-task1-data-segmentation`
3. Enable GPU: Session options → Accelerator → GPU T4 x2
4. Enable Internet: Session options → Internet → ON
5. Upload and run `notebooks/train.ipynb`
6. Download `unet_melanoma.pth` from Output tab

### Option B: Local
1. Download dataset (~14GB):
https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation
2. Place data in:
data/images/   ← ISIC2018_Task1-2_Training_Input
data/masks/    ← ISIC2018_Task1_Training_GroundTruth
3. Create conda environment:
```bash
conda create -n melanoma python=3.10
conda activate melanoma
pip install torch torchvision segmentation-models-pytorch
```
4. Open `notebooks/train.ipynb` and run all cells.

## Next Steps
- Integration with classification model (EfficientNet-B0)
- Full pipeline: detection → segmentation → classification
- Tracking lesions over time