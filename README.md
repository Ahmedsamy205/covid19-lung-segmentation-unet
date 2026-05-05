🫁 COVID-19 Lung Segmentation using U-Net
A deep learning project for automated lung segmentation from chest X-ray images using a U-Net architecture, trained on the COVID-19 Radiography Dataset.

📋 Overview
This project builds a U-Net model to segment lung regions in chest X-ray images across four categories:

COVID-19
Normal
Lung Opacity
Viral Pneumonia

The model takes a grayscale chest X-ray image as input and outputs a binary mask highlighting the lung region.

🗂️ Dataset
COVID-19 Radiography Database — available on Kaggle:
🔗 https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
COVID-19_Radiography_Dataset/
├── COVID/
│   ├── images/
│   └── masks/
├── Normal/
│   ├── images/
│   └── masks/
├── Lung_Opacity/
│   ├── images/
│   └── masks/
└── Viral Pneumonia/
    ├── images/
    └── masks/

🏗️ Model Architecture
Custom U-Net with skip connections:
StageFiltersEncoder Block 164Encoder Block 2128Encoder Block 3256Encoder Block 4512 + Dropout(0.3)Bottleneck512 + Dropout(0.4)Decoder Block 1512Decoder Block 2256Decoder Block 3128Decoder Block 464Output1 (sigmoid)

Input: 256×256 grayscale image
Output: 256×256 binary segmentation mask
Loss: Binary Crossentropy
Optimizer: Adam


⚙️ Requirements
tensorflow
numpy
pandas
pillow
scikit-learn
kaggle

🚀 How to Run
1. Setup Kaggle API
bashmkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
2. Download the Dataset
bashkaggle datasets download -d tawsifurrahman/covid19-radiography-database
unzip covid19-radiography-database.zip
3. Open and Run the Notebook
bashjupyter notebook covid19-lung-segmentation-unet.ipynb
Or run it directly on Google Colab.

🏋️ Training

Image size: 256×256
Batch size: 16
Train/Test split: 80% / 20%
Epochs: up to 5 (with Early Stopping, patience=3)
Best model saved to: best_model.h5


📁 Project Structure
├── covid19-lung-segmentation-unet.ipynb   # Main notebook
├── best_model.h5                          # Saved best model (after training)
├── kaggle.json                            # Kaggle API key (not pushed to GitHub)
├── .gitignore
└── README.md

📝 Notes

kaggle.json contains your API credentials — never push it to GitHub.
The dataset is large (~2GB), download it via the Kaggle API as shown above.
Training was done on Google Colab with GPU acceleration.
