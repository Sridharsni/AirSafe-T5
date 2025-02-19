# ATC Error Correction using T5 Model

## 📌 Project Overview
This project aims to reduce **miscommunication in Air Traffic Control (ATC) communications** by **correcting transcription errors** using a fine-tuned **T5 Transformer Model**. The system takes **noisy transcriptions** from ATC voice recordings and **corrects them** to match standard ATC phraseology, reducing misinterpretations that could lead to **aviation accidents**.

## 🚀 Technologies Used
- **Python** (for scripting and model training)
- **Hugging Face Transformers** (T5 model for error correction)
- **Whisper AI** (for speech-to-text transcription)
- **Pandas & NumPy** (data processing)
- **Matplotlib & Seaborn** (visualization)
- **PyTorch** (model training and inference)
- **BitsAndBytes (8-bit quantization)** (to optimize performance on CPU)
- **JiWER** (for evaluating Word Error Rate - WER)

## 📂 Project Structure
```
ATC_Project/
│── data/                  # Data storage (original & cleaned datasets)
│── models/                # Trained models
│── notebooks/             # Jupyter Notebook version
│── scripts/               # Python scripts for automation
│   │── data_preprocessing.py  # Data cleaning & preparation
│   │── train_model.py         # Model fine-tuning
│   │── evaluate_model.py      # WER analysis & model evaluation
│   │── inference.py           # Model inference for correction
│── README.md             # Project documentation
│── requirements.txt      # Dependencies
```

## 📊 Dataset
- **ATC voice transcriptions** (converted from audio using **Whisper AI**)
- Word Error Rate (WER) analysis of **transcribed vs. actual text**
- Custom dataset for **training T5 to correct transcription errors**

## 🔧 Installation & Setup
### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/ATC_Project.git
cd ATC_Project
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Preprocess Data
```bash
python scripts/data_preprocessing.py
```
### 4️⃣ Train the Model
```bash
python scripts/train_model.py
```
### 5️⃣ Evaluate Model
```bash
python scripts/evaluate_model.py
```
### 6️⃣ Run Inference
```bash
python scripts/inference.py --input "transcribed text needing correction"
```

## 🏆 Results
- **Baseline Word Error Rate (WER):** *~50%*
- **Post-correction WER:** *~15-20%*
- **Model improvements:** *Fine-tuning on ATC-specific data*

## 📌 Future Enhancements
- **Deploy as an API for real-time ATC transcription correction**
- **Train on larger ATC datasets for better domain adaptation**
- **Integrate Large Language Models (LLMs) for context-aware corrections**


