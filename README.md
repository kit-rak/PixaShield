# 🛡️ PixaShield: AI Intelligent Surveillance System

**PixaShield** is a real-time AI-powered surveillance application built with Streamlit and YOLO (v7/v8). It provides anomaly detection through image, video, webcam, or RTSP streams and features smart alerts via email. The system is designed for flexible security use-cases such as intrusion detection, fire recognition, weapon spotting, and more.

## 🚀 Features

- 🔐 **Secure Login Interface** (Streamlit UI)
- 🎥 **Supports Multiple Inputs**: Image, Video, Webcam, RTSP
- 🤖 **YOLOv7 & YOLOv8 Compatibility**
- 📦 **Real-Time Object Detection** with class-wise statistics
- 🎨 **Customizable Bounding Box Colors**
- 📧 **Automated Alert System** via email on detection
- 📊 **Live FPS & Class Frequency Display**
- 🧠 **Dynamic Model Loader** with confidence thresholds

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **Detection Models**: YOLOv7, YOLOv8
- **Computer Vision**: OpenCV, NumPy
- **Backend Utilities**: Pandas, JSON, Threading
- **Email Alerting**: SMTP (via `alert.py`)
- **Device Support**: CPU/GPU & RTSP streams

## 🗂️ Project Structure

```
PixaShield/
├── main.py                # Main Streamlit application
├── model_utils.py         # Helper functions for YOLO detection and UI
├── alert.py               # Email alert system
├── assets/
│   └── logo.png           # App logo
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/PixaShield.git
cd PixaShield
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure to have PyTorch and YOLO (Ultralytics) installed:
```bash
pip install torch torchvision torchaudio
pip install ultralytics
```

## 🧪 Usage

### Run the App

```bash
streamlit run maink.py
```

### Login Credentials (default)

- **Username**: `admin`
- **Password**: `password`

## 🔄 Available Modes

Choose from sidebar options:

- **YOLOv7 / YOLOv8**
- **Input Type**: Image, Video, Webcam, RTSP
- **Confidence Threshold**
- **Color customization** for each class
- **Automatic Email Alerts** with detection frame attached

## Model Training Instructions

To use Pixashield, you need to train your own object detection model. We recommend using **YOLOv7** or **YOLOv8**, which are both powerful, efficient models for real-time object detection.

### Steps for Training:

1. **Set Up YOLOv7 or YOLOv8 Environment**:
   - Follow the official YOLOv7 or YOLOv8 repository for installation and setup instructions.
     - YOLOv7: [YOLOv7 GitHub Repository](https://github.com/WongKinYiu/yolov7)
     - YOLOv8: [YOLOv8 GitHub Repository](https://github.com/ultralytics/yolov8)
   
2. **Prepare Your Dataset**:
   - Collect and annotate your images using a tool like LabelImg or Roboflow.
   - Ensure the dataset is in the required format (e.g., YOLO format with `.txt` annotations).

3. **Training the Model**:
   - With your dataset ready, you can train the model using the provided training scripts in either YOLOv7 or YOLOv8. Below is an example command for training:
     ```bash
     python train.py --data <your_data.yaml> --cfg <your_model_config>.yaml --weights <your_pretrained_weights> --batch-size 16
     ```

4. **Evaluate and Fine-Tune**:
   - Once training is complete, evaluate the performance of your model using the validation dataset.
   - Fine-tune hyperparameters or adjust the model architecture as needed.

5. **Export the Model**:
   - After training, export the model in a format that Pixashield supports (e.g., `.pt` for PyTorch models).

6. **Integrate with Pixashield**:
   - Once your model is trained and exported, you can integrate it into the Pixashield system for object detection tasks.

For additional help, refer to the official YOLOv7 or YOLOv8 documentation for detailed steps and troubleshooting.

## 📧 Email Alert System

Alerts are sent using the `sendmail` function in `alert.py`.  
To enable alerts:
- Configure sender credentials and recipient email in `alert.py`.

## 📄 License

Licensed under the **GNU General Public License v3.0**.  
This means any derivative work must also be open-source and GPL licensed.

## 👨‍💻 Authors

- Kartik Prajapati
- Kunal Pawar
- Ishan Naik
- Ishaan Gupta  
- Team PixaShield | Rajasthan Police Hackathon 1.0 Winners 🏆

## 📬 Contact
- **LinkedIn:** [kit-rak](https://www.linkedin.com/in/kit-rak)
- **GitHub:** [kit-rak](https://github.com/kit-rak)

🚀 **Happy Analyzing!**
