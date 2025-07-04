# Image-Classification-ML

### 🧾 Problem Statement

Plant diseases pose a significant threat to agricultural productivity worldwide. Early and accurate identification of plant diseases is crucial for minimizing crop loss and ensuring food security. Traditional methods of disease detection are often manual, time-consuming, and error-prone.

This project aims to **develop a machine learning-based image classification system** to automatically detect and classify leaf diseases in plants using histogram-based feature extraction. The objective is to train multiple models and identify the one that offers the most reliable performance, which can then be deployed in real-world agricultural monitoring applications.

---

## 📁 Project Structure

| File / Folder           | Description                                      |
|------------------------|--------------------------------------------------|
| `Image_Classification_Project.ipynb` | Main Colab notebook with full ML pipeline     |
| `app.py`                | Streamlit web app for interactive predictions    |
| `final_rf_model.pkl`    | Saved trained Random Forest model (for inference)|
| `requirements.txt`      | Python dependencies                              |
| `README.md`             | Project documentation                            |

---

## 🧪 Dataset Access

The dataset used in this project is hosted on Google Drive.

### 🔗 [Click here to access the dataset folder](https://drive.google.com/drive/folders/1DzqCsDAqHpf5hM-LjmDykjces8JFwQjY?usp=sharing)

### 📌 Instructions for First-Time Users:

1. Open the above link and **add the folder as a shortcut to your Google Drive**.
2. In the Colab notebook, **run the following to mount your drive**:

` ```python `
from google.colab import drive
drive.mount('/content/drive')


## 🧰 ML Models Used

- 📌 **Support Vector Machine (SVM)**
- 🌲 **Random Forest** *(Final Selected Model)*
- 🌟 **Gradient Boosting**
- ⚡ **XGBoost**

---

## 📊 Final Model: Random Forest

- **Accuracy**: 80%  
- **Weighted F1-score**: 0.78  
- ✅ Chosen for its **balanced and stable performance** across all classes.

---

## 🚀 Running the App

To run the **Streamlit app locally**, use the following commands:

` ```bash `
pip install -r requirements.txt
streamlit run app.py

##  License
This project is licensed under the MIT License.


## Acknowledgements
Dataset: Plant Pathology 2020 - FGVC7


