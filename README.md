# CISNet

> Paper Title: CISNet: Change Information Guided Semantic Segmentation Network for Automatic Extraction of Glacier Calving Fronts

---

## Directory Structure
```
project/
├── data_processing/
│ ├── data_postprocessing.py
│ └── data_preprocessing.py
├── dataset/
│ ├── glacier_data.py
│ └── utils.py
├── model/
│ ├── util/
│ │ ├── sync_batchnorm/
│ │ │ ├── batchnorm.py
│ │ │ ├── comm.py
│ │ │ ├── replicate.py
│ │ │ └── unittest.py
│ ├── aspp.py
│ ├── change_head.py
│ ├── cisnet.py
│ └── decoder.py
├── calculate_with_distance.py
├── train.py
├── requirements.txt
└── README.md
```

---
## Environment Setup
```
conda create -n env_name python=3.8
conda activate env_name
pip install -r requirements.txt
```



