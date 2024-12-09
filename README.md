# **Image Demosaicing**

## **Project Overview**
This project demonstrates how to remove mosaic effects from images using machine learning techniques. It implements a **Diffusion Model**, a state-of-the-art generative model, to reconstruct high-quality images from their mosaic-degraded versions. The project is modular, scalable, and can be further extended for other image restoration tasks.

---

## **Features**
- Automatically generates mosaic versions of input images.
- Implements a convolution network with encoder-decoder architecture for image restoration.
- Includes options for visualization and monitoring progress during training.

---

## **Directory Structure**
```plaintext
image_demoisacing/
├── script/
│   ├── data_preparation  # Data preparation script
│   ├── dataset.py        # Data loader implementation
│   ├── model.py          # Model architecture
│   ├── train.py          # Training script for the model
│   ├── utils.py          # Utility functions
├── checkpoints/          # Directory to save model checkpoints
├── requirements.txt      # Python dependencies
└── README.md             # Project description

~/
├── data/
│   ├── original/         # Original clean images
│   ├── mosaic/           # Mosaic-degraded images
```

## Installation

1.	Clone the repository:
```plaintext
git clone https://github.com/your-repo/image-demosaicing.git
cd image_demosaic_project
```
2.	Set up a conda virtual environment (optional but recommended):
```plaintext
conda create -n myenv python=3.9
conda activate myenv
```
Replace `myenv` by your desired name.

3.	Install dependencies:
```plaintext
pip install -r requirements.txt
```

## Usage
1. Prepare the Dataset
* Download the CelebA dataset from Kaggle:
```plaintext
cd ~/data
kaggle datasets download jessicali9530/celeba-dataset
unzip celeba-dataset.zip
```
* Run the data preparation script to generate mosaic-degraded images:
```plaintext
cd image_demoisacing/
python scripts/data_preparation.py
```

2. Train the Model
```plaintext
python scripts/train.py --device=cuda:0
```
The trained model will be saved in the `checkpoints/` directory.

## Future Improvements

* Experiment with advanced U-Net architectures (e.g., multi-scale features).
* Integrate perceptual loss for enhanced image quality.
* Extend the pipeline for other restoration tasks (e.g., denoising, inpainting).

## Acknowledgments

* U-Net: Leveraged as the backbone for noise prediction.

## License

This project is licensed under the MIT License. Feel free to modify and use it for your own purposes.

## Contact

If you encounter issues or have suggestions, please contact:
* Author: Xiangyu MIAO
* Email: your-email@example.com

Enjoy restoring your mosaic-degraded images with cutting-edge AI!