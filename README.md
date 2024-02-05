# Implicit neural obfuscation for privacy preserving medical image sharing
implementation of MIDL2024 submission on implicit neural obfuscation 
by Mattias P. Heinrich (University of Lübeck) and Lasse Hansen (EchoScout.ai)
[Paper page on OpenReview](https://openreview.net/forum?id=Q5CTUZHp5U)

Re-identification is a severe risk to patients' privacy even if proper anonymisation of images is undertaken. Our proposed method advances the state-of-the-art in effective medical image obfuscation strategies with regards to the following three main points:
* robust generative model, by adapting recent work neural implicit representation and compression for video sequences to the obfuscation of a subset of a X-ray collection.
* novel strategy for k-anonymity that only moderately affects visual image quality while substantially reducing re-identification risks
* alleviates the strong requirements of prior work that are based on simultaneous availability of multiple scans per patients at each data provider

The key concept of our proposed implicit neural obfuscation strategy is as follows: A subset of input chest X-rays serve as target for a neural reconstruction decoder that comprises learnable instance embeddings (D-dimensional vector for each data point) and convolutional weights. The reconstructions are supervised with a loss based on SSIM. During inference a k-anonymity mixing is introduced that aims to obfuscate patient information by adding latent code information from other patients.
![concept](https://github.com/mattiaspaul/neuralObfuscation/blob/main/midl2024_neural_obfuscation.png?raw=true)

This repository implements the first two parts (NeRV fitting and Semantic segmentation with obfuscated images). It is in part based on  the exellent work of [NeRV by Haochen](https://github.com/haochen-rye/NeRV) and uses the SSIM implentation of [Evan Su](https://github.com/Po-Hsun-Su/pytorch-ssim).

To install clone the repo and create + activate a new virtualenv ``virtualenv env; source env/bin/activate``
Next get the requirements by runnning ``pip install monai zipfile36 scikit-image tqdm imageio wget torchvision``

Now you can either register for, and download the Kaggle data used in the paper [RANZR CLiP](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data) and preprocess it accordingly (detailed instructions will be provided at a later point) OR you can use a public demo dataset [Montgomery County CXR](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html) using the following command:
```
python create_demo_xray.py
```
Next you can fit the NeRV in three chunks of 48 images and train a SegResNet (CUDA is required):
```
python fit_nerv_ssim.py img_label_demo.pth 0 46
python train_segresnet.py img_label_demo.pth 0 46 8 122 1
```
The implicit neural obfuscation is very quick, the segmentation network may take 15-20 minutes.



