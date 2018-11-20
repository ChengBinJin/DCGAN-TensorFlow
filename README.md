# DCGAN-TensorFlow
This repository is a Tensorflow implementation of Alec Radford's [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, ICLR2016](https://arxiv.org/pdf/1511.06434.pdf).

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/43060132-8929dcea-8e8a-11e8-9203-712cd25141a8.png" width=600)
</p>
  
## Requirements
- tensorflow 1.9.0
- python 3.5.3
- numpy 1.14.2
- pillow 5.0.0
- pickle 0.7.4
- scipy 0.19.0
- matplotlib 2.0.2

## Applied GAN Structure
1. **Generator**
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/43059677-9688883e-8e88-11e8-84a7-c8f0f6afeca6.png" width=700>
</p>

2. **Discriminator**
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/43060075-47f274d0-8e8a-11e8-88ff-3211385c7544.png" width=500>
</p>

## Generated Images
1. **MNIST**
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43060428-dd0b2b88-8e8b-11e8-9f50-e199e1ea22ee.png" width=900>
</p>

2. **CIFAR10**
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43060450-f8a4c2fa-8e8b-11e8-903b-0d7b0086c7fa.png" width=900>
</p>

3. **CelebA**
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43060477-108c7570-8e8c-11e8-854f-4fc7a4da28c3.png" width=900>
</p>

## Documentation
### Download Dataset
MNIST and CIFAR10 dataset will be downloaded automatically if in a specific folder there are no dataset. Use the following command to download `CelebA` dataset and copy the `CelebA' dataset on the corresponding file as introduced in **Directory Hierarchy** information.
```
python download2.py celebA
```

### Directory Hierarchy
``` 
.
│   DCGAN
│   ├── src
│   │   ├── cache.py
│   │   ├── cifar10.py
│   │   ├── dataset.py
│   │   ├── dataset_.py
│   │   ├── download.py
│   │   ├── main.py
│   │   ├── solver.py
│   │   ├── tensorflow_utils.py
│   │   ├── utils.py
│   │   └── dcgan.py
│   Data
│   ├── celebA
│   ├── cifar10
│   └── mnist
```  
**src**: source codes of the DCGAN

### Implementation Details
Implementation uses TensorFlow implementation to train the DCGAN. Same generator and discriminator networks are used as described in [Alec Radford's paper](https://arxiv.org/pdf/1511.06434.pdf), except that batch normalization of training mode is used in training and test mode that we found to get more stalbe results.

### Training DCGAN
Use `main.py` to train a DCGAN network. Example usage:

```
python main.py --is_train=true
```
 - `gpu_index`: gpu index, default: `0`
 - `batch_size`: batch size for one feed forward, default: `256`
 - `dataset`: dataset name for choice [mnist|cifar10|celebA], default: `mnist`
 - `is_train`: training or inference mode, default: `False`
 - `learning_rate`: initial learning rate, default: `0.0002`
 - `beta1`: momentum term of Adam, default: `0.5`
 - `z_dim`: dimension of z vector, default: `100`
 - `iters`: number of interations, default: `200000`
 - `print_freq`: print frequency for loss, default: `100`
 - `save_freq`: save frequency for model, default: `10000`
 - `sample_freq`: sample frequency for saving image, default: `500`
 - `sample_size`: sample size for check generated image quality, default: `64`
 - `load_model`: folder of save model that you wish to test, (e.g. 20180704-1736). default: `None`
 
### Evaluate DCGAN
Use `main.py` to evaluate a DCGAN network. Example usage:

```
python main.py --is_train=false --load_model=folder/you/wish/to/test/e.g./20180704-1746
```
Please refer to the above arguments.

### Citation
```
  @misc{chengbinjin2018dcgan,
    author = {Cheng-Bin Jin},
    title = {DCGAN-tensorflow},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/DCGAN-TensorFlow}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project borrowed some code from [carpedm20](https://github.com/carpedm20/DCGAN-tensorflow)
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.

## Related Projects
- [pix2pix](https://github.com/ChengBinJin/pix2pix-tensorflow)
