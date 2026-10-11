# Rice Variety Classification with CNNs 🌾 🧠

## 🎯 Project Overview
This project builds convolutional neural networks (CNNs) that classify single rice grains from images into five varieties: **Arborio, Basmati, Ipsala, Jasmine and Karacadag**.

Telling rice varieties apart matters for seed quality control, pricing and food labelling, but doing it by eye is slow and inconsistent. Since the varieties differ mainly in grain **shape and size**, this is a natural task for computer vision. The best model reaches **99.15% validation accuracy** with fewer than **30,000 parameters**.

## 🛠️ Tech Stack
- **Language:** Python
- **Deep Learning:** TensorFlow, Keras (Sequential API, EarlyStopping)
- **Libraries:** Pandas, Matplotlib

## 📊 Dataset Description
The [Rice Image Dataset](https://www.muratkoklu.com/datasets/) (Koklu, Cinar & Taspinar, 2021) contains **75,000 RGB images** of single rice grains on a black background:
- **5 classes**, with **15,000 images per variety**, so the dataset is perfectly balanced.
- **Resolution:** 250 × 250 pixels, resized to **64 × 64** for training.
- **Split:** 80% training (60,000 images) and 20% validation (15,000 images), with a fixed seed.

## 🧪 Methodology & Implementation

### 1. Data Pipeline & Exploration
- **Loading:** Used `image_dataset_from_directory` to read images and labels directly from the class folders.
- **Class Distribution:** Counted the images per class in each split from the dataset's file paths, so no images had to be loaded.
- **Visualization:** Displayed four sample grains per variety to inspect the differences in shape.

### 2. Model Architectures
All models share the same building blocks:
- **Rescaling (1/255)** to normalize pixel values to [0, 1].
- **Conv2D → BatchNormalization → ReLU → MaxPooling** blocks for feature extraction.
- **GlobalMaxPooling2D** to reduce each feature map to a single value.
- **Dense (512) → Dropout (0.25) → Dense (5, softmax)** as the classifier head.

| Model | Conv blocks | Input shape | Parameters |
|---|---|---|---|
| Model 1 | 1 | 224 × 224 | 20,485 |
| Model 2 | 2 | 224 × 224 | 29,861 |
| Model 3 | 2 | 64 × 64 | 29,861 |

### 3. Training
- **Optimizer:** Adam (learning rate 1e-4) with sparse categorical cross-entropy.
- **Early Stopping:** Training stops when the validation loss doesn't improve for 4 epochs, and the best weights are restored.
- **Learning Curves:** Plotted training and validation loss per epoch, marking the best epoch.

## 📈 Results

| Model | Best epoch | Val accuracy | Val loss |
|---|---|---|---|
| Model 1 | 20 / 20 | 97.14% | 0.090 |
| Model 2 | 15 / 19 | 99.08% | **0.024** |
| Model 3 | 10 / 14 | **99.15%** | 0.028 |

## 💡 Key Insights
- **Depth Is the Key Factor:** Adding a second convolutional block cut the error rate from 2.9% to about 0.9%, about **3× fewer misclassified grains**, for only about 9,400 extra parameters.
- **Why Depth Helps:** Global max pooling discards spatial position, so each feature has to capture shape on its own. The second block enlarges each feature's view of the image from about 4 × 4 to 10 × 10 pixels, enough to see the grain's curvature and tips.
- **Input Shape Doesn't Matter Here:** Models 2 and 3 differ only in their declared input shape and perform the same, because global max pooling makes the weights independent of image size.
- **No Overfitting:** Training and validation accuracy stay within 0.5 percentage points of each other at the best epoch.
- **Early Stopping Pays Off:** The two-block models' validation accuracy occasionally dips between epochs. Restoring the best weights keeps the 99%+ checkpoint.

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cnn_rice
   ```

2. **Create a virtual environment and install the dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Download the dataset** from [muratkoklu.com/datasets](https://www.muratkoklu.com/datasets/) and extract it so the folder structure looks like this:
   ```
   cnn_rice/
   ├── cnn_rice.ipynb
   ├── requirements.txt
   └── src/
       └── Rice_Image_Dataset/
           ├── Arborio/
           ├── Basmati/
           ├── Ipsala/
           ├── Jasmine/
           └── Karacadag/
   ```

4. **Run the notebook** `cnn_rice.ipynb` in Jupyter or VS Code. A GPU is recommended but not required.

## 📚 Reference
Koklu, M., Cinar, I., & Taspinar, Y. S. (2021). Classification of rice varieties with deep learning methods. *Computers and Electronics in Agriculture*, 187, 106285. https://doi.org/10.1016/j.compag.2021.106285

## Let's Connect
* **LinkedIn**: [Let's network!](https://www.linkedin.com/in/mohammad-amin-saeini)
