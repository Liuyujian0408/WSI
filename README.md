# Learning Whole Slide Image Representations from Few High-Resolution Patches via Cascaded Dual-Scale Reconstruction

This repository contains the implementation of **Cascaded Dual-Scale Reconstruction (CDSR)**, a cascaded dual-scale framework designed for feature extraction from high-resolution whole slide image (WSI) patches. The framework is tailored to extract informative representations from multi-scale views of WSI patches for downstream tasks.

## Project Overview

The project consists of two core models:

- **QHVAE**: A quantized hierarchical variational autoencoder responsible for coarse reconstruction.
- **L2G-Net**: A local-to-global refinement network that enhances fine details based on QHVAE outputs.

Each model is placed in its respective folder:
- `QHVAE/`
- `L2G_Net/`

Each folder also contains:
- `demo.ipynb`: A Jupyter Notebook demonstrating the inference process.
- `c16_fig_recon_test/`: A directory used to test reconstruction quality.

The reconstruction results can be found in the file:
- `QHVAE/c16_fig_recon_test/reconstruction_comparison.jpg`
- `L2G_Net/c16_fig_recon_test/reconstruction_comparison.jpg`

> **Note:** Pretrained `.pt` model files are not included in this repository. The complete codebase will be released upon acceptance of the manuscript.


## Inference Guide

To test the reconstruction quality:
1. Open either `QHVAE/demo.ipynb` or `L2G_Net/demo.ipynb`.
2. Follow the instructions to:
   - Load test patches
   - Run the model for inference
   - Visualize original vs reconstructed patches
   - Compute quality metrics (e.g., PSNR)


