# Implicit neural obfuscation for privacy preserving medical image sharing
implementation of MIDL2024 submission on implicit neural obfuscation 
by Mattias P. Heinrich (University of Lübeck) and Lasse Hansen (EchoScout.ai)
[Paper page on OpenReview](https://openreview.net/forum?id=Q5CTUZHp5U)

Re-identification is a severe risk to patients' privacy even if proper anonymisation of images is undertaken. Our proposed method advances the state-of-the-art in effective medical image obfuscation strategies with regards to the following three main points:
* robust generative model, by adapting recent work neural implicit representation and compression for video sequences to the obfuscation of a subset of a X-ray collection.
* novel strategy for k-anonymity that only moderately affects visual image quality while substantially reducing re-identification risks
* alleviates the strong requirements of prior work that are based on simultaneous availability of multiple scans per patients at each data provider

The key concept of our proposed implicit neural obfuscation strategy is as follows: A subset of input chest X-rays serve as target for a neural reconstruction decoder that comprises learnable instance embeddings (D-dimensional vector for each data point) and convolutional weights. The reconstructions are supervised with a loss based on SSIM. During inference a k-anonymity mixing is introduced that aims to obfuscate patient information by adding latent code information from other patients.
[concept]: https://github.com/mattiaspaul/neuralObfuscation/blob/main/midl2024_neural_obfuscation.png "Concept Figure"
