# SAM-CL
Self-adversarial Multi-scale Contrastive Learning for Semantic Segmentation of Thermal Facial Images

## **Abstract**
Reliable segmentation of thermal facial images in unconstrained settings such as thermal ambience and occlusions is challenging as facial features lack salience. Limited availability of datasets from such settings further makes it difficult to train segmentation networks. To address the challenge, we propose Self-Adversarial Multi-scale Contrastive Learning (SAM-CL) as a generic learning framework to train segmentation networks. SAM-CL framework constitutes SAM-CL loss function and a thermal image augmentation (TiAug) as a domain-specific augmentation technique to simulate unconstrained settings based upon existing datasets collected from controlled settings. We use the Thermal-Face-Database to demonstrate  effectiveness of our approach. Experiments conducted on the existing segmentation networks- UNET, Attention-UNET, DeepLabV3 and HRNetv2 evidence the consistent performance gain from the SAM-CL framework. Further, we present a qualitative analysis with UBComfort and DeepBreath datasets to discuss how our proposed methods perform in handling unconstrained situations.

### **SAM-CL Framework:**
<p align="left">
    <img src="images/SAM-CL%20Framework.png" alt="SAM-CL Framework" width="800"/>
</p>

### **Thermal Image Augmentation:**
<p align="left">
    <img src="images/Thermal%20Augmentation%20Module.png" alt="Thermal Image Augmentation Module" width="600"/>
</p>


### **Demo:**
<p align="left">
    <img src="images/SAM-CL_Demo.gif" alt="Demo of SAM-CL Framework" width="800"/>
</p>

## **Installation**
This implementation is built on [openseg.pytorch](https://github.com/openseg-group/openseg.pytorch) as well as [ContrastiveSeg](https://github.com/tfzhou/ContrastiveSeg).

Please follow the Getting Started of openseg.pytorch for installation and dataset preparation

## **Training**
### **Illustrative command for training segmentation network:**

 Replace the actual paths for all the fields within square brackets (e.g. --config [~/dev/data/ThermalFaceDB])

``` bash
bash scripts/thermalFaceDB/deeplab/run_x_8_deeplabv3_train_samcl_occ.sh train x_8_samcl_occ ~/dev/data/ThermalFaceDB ~/dev/data/ThermalFaceDB
```

## **Inference/ Validation**
### **Illustrative command for running validation:**

 Replace the actual paths for all the fields within square brackets (e.g. --config [~/dev/data/ThermalFaceDB])

``` bash
bash scripts/thermalFaceDB/deeplab/run_x_8_deeplabv3_train_samcl_occ.sh val x_8_samcl_occ ~/dev/data/ThermalFaceDB ~/dev/data/ThermalFaceDB
```

### **Performance Evaluation:**
<p align="left">
    <img src="images/Performance%20Evaluation%20of%20SAM-CL%20Framework.png" alt="Performance Evaluation of SAM-CL Framework" width="800"/>
</p>

### **Ablation Study:**
<p align="left">
    <img src="images/Ablation%20Study%20for%20TiAug%20and%20SAM-CL%20Loss.png" alt="Ablation Study for TiAug and SAM-CL Loss" width="500"/>
</p>

### **Qualitative Analysis  on Different Datasets:**
<p align="left">
    <img src="images/Qualitative%20Outcome%20Combined.png" alt="Qualitative Analysis on Different Datasets" width="800"/>
</p>