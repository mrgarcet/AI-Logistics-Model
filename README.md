# 📦 AI Logistics Model  
NLP + Computer‑Vision + Graph‑Neural‑Network demo
================================================



| Layer | Tech                                                          | Purpose |
|-------|---------------------------------------------------------------|---------|
| **NLP** | *DistilBERT fine‑tune* (<br>`models/bert_intent_classifier/`) | Classify user requests into 5 intents |
| **CV** | *OpenCV + pyzbar* (`src/cv/barcode.py`)                       | Scan barcodes, prompt for ±qty, update inventory |
| **GNN** | *PyTorch‑Geometric GCN‑GRU* (`models/gnn_forecaster/`)        | 30‑day demand forecast |
| **UI** | *Streamlit* (`ui/dashboard.py`)                               | Editable inventory table + chat + scan button |
| **Persistence** | CSV files in `data/`                                          | Survive restarts; easy to inspect |

---

## 1 · Prerequisites

| Requirement | Version                                                                |
|-------------|------------------------------------------------------------------------|
| **Python** | 3.11 (works on 3.10‑3.12)                                              |
| **Git** | latest                                                                 |
| **Webcam** | any UVC cam (Razer Kiyo Pro used here)                                 |
| **Windows privacy** | *Settings ▷ Privacy ▷ Camera → Let desktop apps access your camera ON* |

---
## 🚀 Quick-start (local install)

> Follow these steps after cloning the repository  
> `git clone https://github.com/<your-org>/AI_Logistics_Model.git`
2 Install core Python packages
PyTorch and PyTorch-Geometric are pinned to CPU wheels.
If you have a CUDA GPU, replace the first command with the wheel from https://pytorch.org/get-started/locally/.

bash
Copy
Edit
pip install torch==2.2.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric==2.5.2 -f https://data.pyg.org/whl/torch-2.2.1+cpu.html
pip install -r requirements.txt          # transformers, streamlit, opencv-python-headless, etc.
3 Install ZBar (barcode backend)

OS	How
Windows	Download the official DLL bundle https://github.com/NaturalHistoryMuseum/ZBarWin64/releases → unzip → copy libzbar-64.dll into %VENV%\Lib\site-packages\pyzbar\
macOS	brew install zbar
Linux	sudo apt install libzbar0 (Debian/Ubuntu)
4 Download YOLOv8n weights (≈ 8 MB)
powershell
Copy
Edit
# run from the project root
Invoke-WebRequest `
    -Uri "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt" `
    -OutFile "src\cv\weights\yolov8n.pt"
(Bash users can use curl -L -o src/cv/weights/yolov8n.pt https://github.com/...)
