# PyraMal: Byte-level Malware Detection and Classification via Pyramid Feature Map

**PyraMal** is a novel, platform-agnostic malware detection and classification framework that leverages **raw binary bytes** of executable files (e.g., APK, PE) to construct **pyramid-structured feature maps**, enabling high-accuracy, robust, and generalizable malware analysis without relying on parsing internal structures (e.g., AndroidManifest.xml, API calls, or dex files).

> 📌 **Key Advantages**
>
> - Works directly on raw binary data — no parsing required
> - Strong cross-dataset generalization
> - Achieves **99.94% accuracy** on the MalImg dataset and **99.60% accuracy** on the BIG-2015 dataset
> - Compatible with Android APK and Windows PE formats

------



## 📊 Performance Highlights

| Dataset                | Task                             | Accuracy   | Recall | Precision | F1-Score |
| ---------------------- | -------------------------------- | ---------- | ------ | --------- | -------- |
| **CICMalDroid2020Det** | Malware Detection                | **98.48%** | 98.05% | 97.74%    | 97.89%   |
| **BIG-2015**           | Malware Classification           | **99.60%** | 99.57% | 99.55%    | 99.56%   |
| **CICMalDroid2020Cls** | 4-Class Classification           | **95.79%** | 94.79% | 94.64%    | 94.69%   |
| **MOTIF-TOP100**       | PE Malware Family Classification | **95.12%** | 94.88% | 94.75%    | 94.80%   |

*(See paper for full results and confusion matrices)*

------

## 🛠️ Installation

### Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 1.12
- NumPy, scikit-learn, matplotlib, tqdm

### Setup



```
1git clone https://github.com/yourname/pyramal.git
2cd pyramal
3pip install -r requirements.txt
```

------

## 🚀 Quick Start

### 1. Prepare Dataset





```
1data/
2├── train/
3│   ├── malware/
4│   └── benign/
5└── test/
6    ├── malware/
7    └── benign/
```

> 💡 Supported formats: `.apk`, `.exe`, `.dll`, or any raw binary file.

### 2. Train the Model



```
1python train.py \
2  --data_dir ./data/train \
3  --model_save_path ./models/pyramal.pth \
4  --epochs 50 \
5  --batch_size 64
```

### 3. Evaluate / Predict









```
1python evaluate.py \
2  --model_path ./models/pyramal.pth \
3  --test_dir ./data/test
```

Or classify a single file:





```
1python predict.py --file sample.apk --model ./models/pyramal.pth
2# Output: MALWARE (confidence: 0.987)
```

------

## 🧠 Method Overview

PyraMal converts raw binary sequences into **multi-scale pyramid feature maps** that preserve structural patterns across byte-level, block-level, and file-level granularities. This avoids:

- Parsing failures in obfuscated/malformed samples
- Dependency on volatile features (e.g., API calls, permissions)
- Image distortion from fixed-size resizing

The model uses a lightweight CNN backbone optimized for pyramid inputs, ensuring efficiency and scalability.



![PyraMal Framework](ReadMe.assets/framework.png)

*Figure: Overview of the PyraMal pipeline (see paper Fig. 2)*



------

## 📄 License

This project is licensed under the **MIT License** — see LICENSE for details.

------

## 🙌 Acknowledgements

- Datasets: CICMalDroid2020, BIG-2015, MOTIF
- Inspired by vision-based malware analysis, but designed for **robustness over convenience**

------

## ❓ Questions or Contributions?

Feel free to open an issue or submit a pull request! We welcome:

- New dataset support
- Model optimizations
- Visualization tools
- Docker deployment scripts

------

> ⚠️ **Disclaimer**: This tool is for **academic and defensive cybersecurity research only**. Do not use on systems you do not own or without proper authorization.