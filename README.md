# Multimodal Movie Genre Classification

## 🌐 Website Link

**Netlify Deployment**: [Visit Website](https://genregenie.netlify.app/)
This site presents our project output, including model performance and results.


---
# ROLES AND RESPONSIBILITIES

### DATA SPECIALIST AND IMAGE MODELLING LEAD: ANUPAM PATIL
### FUSION AND TEXT MODELLING LEAD: NANDINI SONI

---

## Overview

This project focuses on classifying movie genres by combining both **textual** and **visual** features in a multimodal deep learning pipeline. We experimented with:

* Extracting **textual features** from movie plot overviews using an LSTM-based model.
* Extracting **visual features** from movie posters using precomputed embeddings (EfficientNet/ResNet).
* **Fusing both modalities** using attention-based mechanisms to improve genre classification accuracy.

We worked with a cleaned and undersampled version of the TMDB dataset containing around 20,530 movie samples, focusing on 12 major genres.

---
## Repository Structure

* `Data_Collection_EDA_Specialist.ipynb`: Data fetching, preprocessing, genre filtering, and balancing operations.
* `LSTM_Text_Model.ipynb`: Text-only model using a 2-layer BiLSTM for genre classification from overviews.
* `Image_Resnet_Model.ipynb`: Trains a ResNet-based model on poster images and generates 1280-dimensional embeddings.
* `Fusion_Model_LSTM_RESNET.ipynb`: Combines text and image features using attention for final genre predictions.
* `poster_embeddings.pt`: Poster features generated using EfficientNet and projected to 512 dimensions for fusion.
* `undersampled_movie20530.csv`: Final cleaned dataset used in the models.
* `README.md`: This documentation file.

---

## Models Used

### Text Model

* 2-layer BiLSTM
* Embedding Dim: 128
* Hidden Dim: 256
* Output: 256-dim text representation

### Image Model

* Precomputed EfficientNet/ResNet embeddings (1280-dim)
* Linear layer to reduce to 512-dim
* ReLU + Dropout (0.3) for regularization

### Fusion Model

* Attention-based fusion using Multihead Self-Attention
* Stacks 512-dim text and image features
* Averages the attention output
* Final classification with Linear(512 → 12) for multi-label genre prediction

---

## Training Setup

* **Loss Function**: `BCEWithLogitsLoss()` (suitable for multi-label)
* **Optimizer**: `AdamW` with L2 regularization (`weight_decay`)
* **Epochs**: 20
* **Batch Sizes**: 16, 32, and 64 tested
* **Learning Rates**: 0.001, 0.0004 (EfficientNet); 2e-5 (Fusion)
* **Scheduler**: CosineAnnealingLR (T\_max = 20)
* **Dropout**: Applied (rate = 0.3) to prevent overfitting

---

## Data Details and Preprocessing

* Movie metadata was collected using TMDB API.
* Posters were downloaded and locally saved.
* Genre filtering was applied to focus on top 12 popular genres.
* Used time-based diversity by including movies from two periods: 1975–2000 and 2001–2025.
* Performed one-hot encoding for genre columns.
* Tokenized overviews for LSTM, and normalized image inputs for CNN-based feature extraction.

---

## Evaluation Metrics

* **Exact Match Accuracy**: Percentage of samples where all predicted genres exactly match true genres.
* **Micro F1 Score**: Captures overall performance across all genres.
* **Confusion Matrix**: Analyzed for key genres like Drama, Action, Comedy.

---

## Results Summary

* **Text-Only Model (LSTM)**: \~41.5% train accuracy after 10 epochs; lower generalization on validation.
* **Image-Only Model (EfficientNet)**: Better generalization (\~50% accuracy) with focused dropout.
* **Fusion Model (LSTM + Image Embeddings + Attention)**: Best performance with attention-based fusion using self-attention, reaching \~62% train and \~50% validation accuracy.

---

## Getting Started

To run this project locally or in Colab:

1. Clone the repository:

   ```bash
   git clone https://github.com/patilanupam/Multimodal-Movie-Classification.git
   cd Multimodal-Movie-Classification
   ```

2. Install the necessary packages:

   * PyTorch
   * scikit-learn
   * transformers (for BERT variants if needed)
   * torchvision
   * matplotlib, pandas, numpy

3. Run the notebooks in order:

   * `Data_Collection_EDA_Specialist.ipynb`
   * `LSTM_Text_Model.ipynb`
   * `Image_Resnet_Model.ipynb`
   * `Fusion_Model_LSTM_RESNET.ipynb`

---

## Final Remarks

* The attention-based fusion model significantly outperformed the individual text-only and image-only models.
* Preprocessing, especially genre filtering and poster quality, played a key role in improving generalization.
* The project demonstrates the strength of combining multimodal data for robust genre classification.

---

## Research Papers

* A Multi-Attention Approach Using BERT and Stacked Bidirectional LSTM for Improved Dialogue State Tracking
by Muhammad Asif Khan 1,†ORCID,Yi Huang 2,Junlan Feng 2,Bhuyan Kaibalya Prasad 3,ORCID,Zafar Ali 1,Irfan Ullah 4,ORCID andPavlos Kefalas 5
* Attention Is All You Need
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
* Deep Residual Learning for Image Recognition 
by Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun
Microsoft Research
{kahe, v-xiangz, v-shren, jiansun}@microsoft.com
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
Mingxing Tan 1 Quoc V. Le 1

