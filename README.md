# Image Processing Strategies on Ensemble Models

A hybrid deep learning framework for **robust malaria cell image classification** using **CNN–Transformer ensemble architectures** combined with **advanced data augmentation strategies**.

---

## 📌 Overview

This project investigates how **heterogeneous ensemble models**, combining Convolutional Neural Networks (CNNs) and Transformer-based architectures, can improve **accuracy, robustness, and calibration** in medical image classification tasks.

Using the **NIH Malaria Cell Dataset**, we evaluate three dual-backbone ensemble architectures under multiple augmentation strategies. The study demonstrates that **feature-level fusion combined with MixUp augmentation** significantly outperforms single-model baselines.

---

## 🧠 Key Contributions

- Designed **three hybrid CNN–Transformer ensemble architectures**
- Implemented **late fusion at the embedding level**
- Compared **MixUp** and **Mixture-of-Augmentations**
- Improved generalization while reducing overfitting
- Identified the **DenseNet121 + Swin Transformer** ensemble as the most stable and accurate model

---

## 🏗️ Ensemble Architectures

Each ensemble consists of **two independently fine-tuned backbones**, each producing a **512-dimensional feature embedding**.  
The embeddings are concatenated into a **1024-dimensional fused representation**.

### Model Combinations

1. **ResNet50 + Vision Transformer (ViT)**  
   Combines CNN-based texture extraction with global self-attention.

2. **DenseNet121 + Swin Transformer** ⭐  
   Dense feature reuse combined with hierarchical windowed attention.

3. **EfficientNetB3 + ConvNeXt-Tiny**  
   Modern CNN architectures optimized for efficiency and performance.

---

## 🔀 Feature Fusion Strategy

- **Late Fusion** via embedding concatenation  
- Fusion classifiers:
  - Logistic Regression (interpretable baseline)
  - Fully Connected Neural Network (non-linear fusion)

This design preserves complementary feature representations while remaining modular and extensible.

---

## 🧪 Data Augmentation Strategies

Augmentations are applied **only to the training set** to maintain unbiased evaluation.

### 1️⃣ MixUp Augmentation
- Linearly combines image pairs and their labels
- Smooths decision boundaries
- Reduces overfitting and memorization
- Produced the **highest generalization gains**

### 2️⃣ Mixture-of-Augmentations
At each iteration, one augmentation is randomly applied:
- Rotation + Zoom
- Brightness / Contrast / Saturation adjustment
- Horizontal / Vertical flips
- Gaussian noise injection
- Random crop + padding

---

## 📊 Dataset

- **Source:** NIH Malaria Dataset  
- **Total Images:** ~27,559  
- **Classes:** Infected / Uninfected  
- **Image Type:** Microscopy cell images  
- **Preprocessing:** Resize (224×224), normalization, one-hot encoding

---

## 📈 Experimental Results (Test Accuracy)

| Ensemble | Base Dataset | Mixture Augmentation | MixUp |
|--------|-------------|----------------------|-------|
| ResNet50 + ViT | 72.6% | 50.5% | 72.8% |
| **DenseNet121 + Swin** | **92.6%** | **92.2%** | **92.2%** |
| EfficientNet + ConvNeXt | 69.3% | 67.1% | 62.8% |

**DenseNet121 + Swin Transformer** consistently delivers the best performance and stability.

---

## 🔍 Key Insights

- CNN–Transformer synergy is crucial for medical imaging
- MixUp is more stable than aggressive spatial augmentations
- Backbone compatibility matters more than individual model strength
- Late fusion improves robustness and calibration
- Ensembles outperform single architectures across all settings

---

## 🧰 Tech Stack

- **Frameworks:** TensorFlow, Keras  
- **Models:** ResNet50, DenseNet121, EfficientNetB3, ViT, Swin Transformer, ConvNeXt  
- **Libraries:** NumPy, OpenCV, Albumentations  
- **Training:** Partial fine-tuning, soft-label learning  
- **Evaluation:** Accuracy, Loss, Comparative Stability Analysis  

---

## 🚀 Future Work

- Integrate **Explainable AI** (Grad-CAM, Attention Rollout)
- Apply **Neural Architecture Search (NAS)** for optimal ensemble selection
- Use **Knowledge Distillation** for edge deployment
- Replace static fusion with **adaptive gated attention fusion**

---

## 👨‍💻 Authors

- **Kaushik Roy** – DenseNet121 + Swin Transformer  
- **Malay** – ResNet50 + Vision Transformer  
- **Anvesh Chandrakar** – EfficientNetB3 + ConvNeXt-Tiny  
- **Garima Khurana** – EfficientNetB3 + ConvNeXt-Tiny  

**Institution:** KIIT Deemed to be University  
**Department:** School of Computer Engineering  
**Year:** 2025  

---

## 📜 License

This project is intended for **academic and research purposes**.  
Please cite the original work if reused or extended.
