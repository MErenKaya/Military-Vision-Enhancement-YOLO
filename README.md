# Military Vision Enhancement & Object Detection Pipeline

[![C++](https://img.shields.io/badge/C%2B%2B-11%2F17%2F20-blue.svg)](https://en.cppreference.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX-orange.svg)](https://github.com/ultralytics/ultralytics)

A high-performance C++ computer vision pipeline designed for challenging, low-visibility environments (fog, haze, dust). This system integrates a lightweight **AOD-Net** image enhancement model—which I trained—with **YOLOv8** for real-time military object detection.

---

## Key Features

* **Two-Stage Adaptive Pipeline:** Integrates 9 KB ultra-lightweight **AOD-Net** for real-time dehazing and contrast enhancement prior to object detection.
* **Optimized C++ Implementation:** Built natively using **OpenCV DNN module** (`DNN_BACKEND_OPENCV`) running entirely on CPU with efficient memory management (no heavy Python runtime overhead).
* **Robust Military Detection:** Trained/adapted to detect critical classes (*military tanks, armored vehicles, camouflage soldiers, trenches, etc.*) even under extreme weather degradation.
* **Before / After Comparison Support:** Includes built-in side-by-side visualization to analyze how dehazing improves confidence scores and reduces false alarms in chaotic environments.

---

## Performance & Results

Under heavy atmospheric degradation (fog/dust storms), the AOD-Net preprocessing layer yields an **approx. 10–15% accuracy and reliability boost** over standard raw inference by recovering hidden edge details.

Under heavy atmospheric degradation (fog/dust storms), the AOD-Net preprocessing layer yields an **approx. 10–15% accuracy and reliability boost** over standard raw inference by recovering hidden edge details.

* **Left:** Original Hazy Input (Degraded Vision) | **Right:** AOD-Net Enhanced Pipeline Output
![Before After Comparison](result.png)

---

##  Tech Stack & Architecture

* **Language:** C++
* **Computer Vision & Inference:** OpenCV (DNN module, `blobFromImage`, `NMSBoxes`, Letterbox scaling)
* **Models:** 
  * `aodnet_480x640.onnx` (Image restoration / Dehazing)
  * `yolo8n_military.onnx` (Object detection)

---

## ⚙️ Getting Started

### Prerequisites
* OpenCV 4.x installed and configured with Visual Studio / C++ compiler.
* ONNX model weights (`aodnet_480x640.onnx` and `yolo8n_military.onnx`).

### Build & Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/MErenKaya/Military-Vision-Enhancement-YOLO.git](https://github.com/MErenKaya/Military-Vision-Enhancement-YOLO.git)
