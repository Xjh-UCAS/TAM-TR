<h2> 
<a href="https://github.com/Xjh-UCAS/TAM-TR/" target="_blank">TAM-TR: Text-guided Attention Multi-Modal Transformer for Object Detection in UAV Images</a>
</h2>

This is the official PyTorch implementation of the following publication:

> **TAM-TR: Text-guided Attention Multi-Modal Transformer for Object Detection in UAV Images**<br/>
> *Elsevier ISPRS 2025*<br/>
> [**Paper**](https://doi.org/10.1016/j.isprsjprs.2025.04.027) | [**Project-page**](https://github.com/Xjh-UCAS/TAM-TR/) 


## 🔭 Introduction
<p align="center">
<strong>TAM-TR: Text-guided Attention Multi-Modal Transformer for Object Detection in UAV Images</strong>
</p>
<img src="https://github.com/user-attachments/assets/c5fdc8f7-3530-430a-ab36-ef4ca843fadd" alt="Motivation" style="zoom:25%; display: block; margin-left: auto; margin-right: auto; max-width: 100%;">

<p align="justify">
<strong>Abstract:</strong> —Object detection in unmanned aerial vehicles (UAVs) imagery is crucial in many fields, such as
maritime search and rescue, remote sensing mapping, urban management and agricultural monitoring.
The diverse perspectives and altitudes of UAV images often result in significant variations in
the appearance and dimensions of objects, and occlusion occurs more frequently in UAV images
than in natural scenes. The unique bird’s-eye view of drones makes it more difficult for existing
object detection models to distinguish between similar objects. A text-guided attention multi-modal
transformer network named TAM-TR is proposed to address the above challenges. A Bidirectional
Text-Image Attention Path Aggregation Network (BTA-PAN) is proposed in TAM-TR, which imitates
the architecture of the classic algorithm Scale-Invariant Feature Transform (SIFT) and is successfully
applied to deep learning models after being improved, showing better scale adaptability. A novel
Multi-modal encoder-decoder head (MEH) was proposed, which can simultaneously consider all
input sequence positions to avoid the disappearance of features of occluded objects. An additional
text-guided attention branch, combined with a large text model, was proposed to enhance the
TAM-TR model’s classification performance. Additionally, a novel Rotation-invariant IOU (RIOU)
loss function was proposed to eliminate the rotational instability in the previous loss function.
Experimental results demonstrate that our TAM-TR model achieves 39.7% mean averaged precision
(mAP) on the Visdrone dataset, surpassing the baseline by 9.5%.
</p>

## 💻 Installation
🔴 Note: Due to the use of the Vmanba library, this code can only be run on linux devices:
The Vmanba library installation refers to this connection：
https://github.com/MzeroMiko/VMamba

pip install -r requirements.txt

### Training
Example: train ```TAM-TR``` on the Visdrone2019 dataset
```
python trainTAMTR.py
```

### Evaluation
Example: val ```TAM-TR``` on the Visdrone2019 dataset
```
python valTAMTR.py
```

### Predict
Example: predict ```TAM-TR``` on the Visdrone2019 dataset
```
python predictTAMTR.py
```

## 💡 Citation
If you find this repo helpful, please give us a star~.Please consider citing Mobile-Seed if this program benefits your project.
```
@article{XU2025170,
title = {TAM-TR: Text-guided attention multi-modal transformer for object detection in UAV images},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {227},
pages = {170-184},
year = {2025},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2025.04.027},
url = {https://www.sciencedirect.com/science/article/pii/S0924271625001637},
author = {Jianhao Xu and Xiangtao Fan and Hongdeng Jian and Chen Xu and Weijia Bei and Qifeng Ge and Teng Zhao and Ruijie Han},
keywords = {Object detection, UAV image, Multi-modal, Loss function, Transformer}
}
```

## 🔗 Related Projects
We sincerely thank the excellent projects:
- [Yolov9](https://github.com/WongKinYiu/yolov9) 
- [ultralytics](https://github.com/ultralytics/ultralytics) 
- [Vmamba](https://github.com/MzeroMiko/VMamba) 
