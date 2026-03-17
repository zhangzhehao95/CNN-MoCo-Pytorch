# CNN-MoCo-Pytorch

Pytorch code for paper “Deep learning-based motion compensation for four-dimensional cone-beam computed tomography (4D-CBCT) reconstruction”.

## Citation
Zhang, Z, Liu, J, Yang, D, Kamilov, US, Hugo, GD. Deep learning-based motion compensation for four-dimensional cone-beam computed tomography (4D-CBCT) reconstruction. Med Phys. 2022; 1- 13. https://doi.org/10.1002/mp.16103

## Prerequisites 
* scikit-image==0.25.0
* scipy==1.14.1
* SimpleITK==2.4.0
* tensorboard==2.18.0
* torch==2.5.0
* torchaudio==2.5.1
* torchinfo==1.8.0
* torchvision==0.20.1
* tqdm==4.67.1
  
## Pretrained weights
https://drive.google.com/drive/folders/1FL8zqCIHvYtF9DTJxUJ_2vVKhufLVsGb?usp=drive_link

After getting the pretrained weights, run the code: 
```
python code/main.py -c configs/test_demo.yaml
```

## Tensorflow version
https://github.com/zhangzhehao95/CNN-MoCo
