
---

# 🧠 Skin Disease Detection using PyTorch

## 📖 Overview

This project is a **deep learning-based web application** for detecting and analyzing various **skin diseases** from medical images. It uses **PyTorch** as the main AI framework to perform multi-task classification — predicting both the **type of disease** and its **severity level**. The app also generates **Grad-CAM heatmaps** to visually explain which regions of the image influenced the model’s decision.

---

## 🚀 Key Features

* 🧬 **AI-powered skin disease prediction** using Vision Transformer (ViT)
* 🔥 **Grad-CAM visualization** to highlight important image regions
* 🧠 **Multi-task learning** – predicts both disease type and severity
* ⚡ **GPU acceleration (CUDA)** for faster inference
* 🌐 **Flask web interface** for uploading and analyzing images
* 💾 **Automatic image saving** for original and processed images

---

## 🧰 Technologies Used

| Component     | Technology                     |
| ------------- | ------------------------------ |
| Deep Learning | **PyTorch**, TorchVision       |
| Model         | **Vision Transformer (ViT)**   |
| Backend       | **Flask (Python)**             |
| Frontend      | HTML, CSS, JavaScript          |
| Visualization | Grad-CAM                       |
| Deployment    | CUDA-enabled GPU (NVIDIA A100) |

---

## 🧩 Project Structure

```
skindiseaseproject/
│
├── app.py                # Flask application entry point
├── model.py              # PyTorch model (Multi-task ViT)
├── gradcam.py            # Grad-CAM visualization script
├── static/               # Stores uploaded & Grad-CAM images
├── templates/            # HTML templates for web pages
└── requirements.txt      # Required Python libraries
```

---

## ⚙️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Flask App

```bash
python app.py
```

Optionally, specify a custom port:

```bash
python app.py --port=8081
```

### 3️⃣ Open in Browser

Go to: **[http://localhost:8080/](http://localhost:8080/)** or the port you set.

---

## 🧠 About PyTorch in This Project

PyTorch is the **core deep learning engine** used here. It:

* Loads the trained Vision Transformer model
* Processes the uploaded skin images
* Runs predictions to identify diseases and severity levels
* Generates Grad-CAM heatmaps for interpretability

In short, PyTorch acts as the **“AI brain”** that powers all the prediction and visualization capabilities of this system.

---

## 📸 Example Output

* **Original Image:** Saved in `static/original/`
* **Grad-CAM Image:** Saved in `static/gradcam/`
* **Predicted Output:** Shown on the result page (disease + severity)

---


## 📄 License

This project is developed for academic and research purposes. All rights reserved © 2025.

---

