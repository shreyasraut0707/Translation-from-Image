# Translation from Image

A machine learning based application that extracts English text from images and videos and translates it to Hindi.

## Project Overview

This project implements an OCR (Optical Character Recognition) system combined with language translation. It can:

- Extract text from images using EasyOCR and Tesseract
- Process video files to extract text from frames
- Translate extracted English text to Hindi
- Detect and filter non-English text

## Features

- Text extraction from images using EasyOCR and Tesseract OCR
- English to Hindi translation using Google Translate
- Support for multiple image formats (PNG, JPG, JPEG, BMP, TIFF)
- Video processing support (MP4, AVI, MOV, MKV)
- User-friendly graphical interface built with Tkinter
- Custom trained CRNN model for OCR (trained on IIIT5K dataset)

## Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on system
- Internet connection (required for Google Translate)

## Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/shreyasraut0707/Translation-from-Image.git
cd Translation-from-Image
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Tesseract OCR

For Windows:

- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install and add to system PATH

For Linux:

```bash
sudo apt-get install tesseract-ocr
```

For macOS:

```bash
brew install tesseract
```

### Step 5: Run the Application

```bash
python app.py
```

This will open the GUI window where you can:

1. Click "Choose File" to select an image or video
2. Click "Extract" to extract text from the file
3. Click "Translate" to translate the extracted text to Hindi

## Project Structure

```
Translation from Image/
|-- app.py                  # main application entry point
|-- config.py               # configuration settings
|-- requirements.txt        # python dependencies
|-- README.md               # this file
|-- models/
|   |-- crnn_model.h5       # trained CRNN model
|   |-- best_weights.h5     # model weights
|-- src/
|   |-- ocr_engine.py       # OCR text extraction
|   |-- translator.py       # translation module
|   |-- video_processor.py  # video frame processing
|   |-- data_loader.py      # dataset loading
|   |-- gui/
|   |   |-- main_window.py  # main GUI window
|   |   |-- components.py   # GUI components
|   |-- models/
|       |-- crnn_model.py   # CRNN model architecture
|       |-- train.py        # model training script
```

## Technologies Used

- TensorFlow/Keras - deep learning framework
- OpenCV - image processing
- EasyOCR - optical character recognition
- Tesseract OCR - text extraction
- Google Translate - language translation (free, no API key)
- Tkinter - GUI framework
- IIIT5K Dataset - for training the CRNN model

## Custom Trained Model (Optional)

The application works out-of-the-box using EasyOCR and TrOCR. However, you can optionally train a custom CRNN model:

### Training the Custom Model

1. Download the IIIT5K dataset from: http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html
2. Extract to `IIIT5K-Word_V3.0/` folder in the project root
3. Run the training script:

```bash
python src/models/train.py --epochs 50 --batch_size 32
```

The trained model will be saved to `models/` directory.

### Model Training Results

The CRNN model was trained on the IIIT5K dataset:

- Training images: 2000
- Test images: 3000
- Training accuracy: 76.93%
- Validation accuracy: 75.72%

To retrain the model:

```bash
python src/models/train.py --epochs 10 --batch_size 16
```

## Troubleshooting

Tesseract not found error:

- Verify Tesseract is installed correctly
- Add Tesseract path to system PATH variable

Translation not working:

- Check internet connection
- Verify the text is in English

No text detected:

- Ensure image has clear, readable text
- Try with higher resolution images

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- Minimum 4GB RAM
- GPU optional (for faster processing)

## License

This project is developed for educational purposes.
