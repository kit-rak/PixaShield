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
├── main.py               # Main Streamlit application
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
