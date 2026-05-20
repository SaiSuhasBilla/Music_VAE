# Music_VAE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaiSuhasBilla/Music_VAE/blob/main/VAE.ipynb)


#  1D-ResNet CVAE Audio Generation Pipeline


An advanced deep-learning audio synthesis engine that utilizes a **Convolutional Variational Autoencoder (CVAE)** structurally reinforced with **1D Residual Blocks (ResNet)** to reconstruct and generate raw waveforms. Optimized for 30-second high-accuracy audio sequencing, this model maps complex acoustic features into a low-dimensional continuous latent space ($2\text{D}$) to generate distinct musical genres like Jazz and Classical.

---

##  Interactive Live Playback

>  **Note on Audio Output:** Due to browser security restrictions, static GitHub notebook previews cannot play audio files out loud. To interact with the live audio players, generate fresh tracks, and see the real-time training plots, click the **Open in Colab** badge above to launch the code in a live cloud sandbox.

---

## 📊 Core Architectural Specifications

The engine bypasses heavy spectral conversions (like Mel-spectrogram processing) and trains directly in the **time-domain over raw audio matrices**.

| Parameter | Configuration Specification |
| :--- | :--- |
| **Audio Target Duration** | 30.0 Seconds |
| **Native Sampling Rate ($f_s$)** | $3000\text{ Hz}$ |
| **Input Sequence Array Shape** | $1 \times 90,001$ Floating-point matrix samples |
| **Latent Space Dimensions** | $2\text{D}$ (Continuous Distribution Gaussian Tracking) |
| **Batch Optimization Size** | 10 Tracks per step |
| **Epoch Iterations** | 20 (Stratified matrix training) |

---

##  Deep Learning Architecture Layout

The network uses a symmetric encoder-decoder design optimized using the Keras Functional API. It relies on 1D Residual skip connections to eliminate gradient degradation across deep audio dimensions.

### 1. Encoder Blueprint
* **Input Layer:** Accepts a $1 \times 90,001$ raw audio signal tensor.
* **Feature Extraction:** 4 sequential stages of 1D Convolutional layers interlocked with custom `resnet_1d_block` modules, scaling filter depths from $64 \rightarrow 128 \rightarrow 256$.
* **Normalization:** Native `LayerNormalization` arrays to stabilize internal activations across lengthy temporal step structures.
* **Bottleneck Split:** Flattens spatial data down into twin dense linear layers tracking Mean ($\mu$) and Log-Variance ($\log \sigma^2$).

### 2. Decoder Blueprint
* **Reparameterization Trick:** Samples a random noise tensor $\epsilon \sim \mathcal{N}(0, I)$ to smoothly transition latent code coordinates into viable structural arrays: $z = \mu + \epsilon \odot \exp(\frac{1}{2}\log \sigma^2)$.
* **Symmetric Reconstruction:** Upscales latent dimensions back out through mirror-aligned 1D Transposed Convolution networks (`Conv1DTranspose`), feeding into Batch Normalization pipelines.
* **Output Signal:** Outputs a fully formed $1 \times 90,001$ waveform mapping.

---

##  Mathematical Stability Engineering

To combat the traditional "NaN Gradient Explosion" common when calculating continuous time-domain probabilities, this framework integrates two critical protective guards:

1. **Logit Clamping Boundary:** Custom loss elements pass reconstructed outputs through an active boundary constraint (`tf.clip_by_value(x_logit, -20.0, 20.0)`). This prevents the sigmoid cross-entropy loss equations from attempting to process infinity vectors or computing invalid $\log(0)$ values.
2. **Global Norm Gradient Clipping:** Optimizer pipelines wrap back-propagation chains inside a global threshold guard (`tf.clip_by_global_norm(gradients, 1.0)`). This tames sudden loss spikes, keeping training metrics stable across all 20 epochs.

---

##  Quick-Start Pipeline Guide

The entire dataset handles streaming cloud fetches dynamically. You can replicate this project inside any blank Python workspace using these sequential notebook execution steps:

### 1. Cloud Infrastructure Download
Pulls the original audio dataset into your cloud storage and extracts file trees instantly:
```python
!curl -L -o /content/genres_original.zip "[https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification)"
