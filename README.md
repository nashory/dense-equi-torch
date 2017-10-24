# dense-equi-torch
Torch7 implementation of ["Unsupervised object learning from dense equivariant image labelling"](https://arxiv.org/abs/1706.02932)

___Note: I am working on training/test regressor code to make cleaner version.___  
___but pretraining the network for latent space mapping works prefectly now. (don't worry :))___


# Prerequisites
+ Torch7
+ [thinplatspline](https://github.com/olt/thinplatespline)
+ python 2.7
+ other torch packages (xlua, display, hdf5, image ...)

~~~
loarocks install display
loarocks install hdf5
loarocks install image
loarocks install xlua
~~~



# Usage
first, download CelebA dataset [(here)](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8).
~~~
<data_path>
           |-- image 1
           |-- image 2
           |-- iamge 3 ...
~~~

To train the feature extractor(CNN):
~~~
1. change options in "script/opts.lua" and "data/gen_tps.py"
2. do "th pretrain.lua"
>> pretrained model will saved in 'repo/pretrain/'
~~~

To train the regressor(mlp):
~~~
1. change options in "script/opts.lua" and "data/gen_reg.lua"
2. do "th regtrain.lua"
>> trained regressor will saved in 'repo/regressor/'
~~~

To test the regressor(mlp):
~~~
1. change options in "script/opts.lua"
2. do "th regtest.lua"
>> test image wih landmarks will be saved in 'repo/test'
~~~


# Results

### (1) mapping on the latent space
  + Red : left-mouth
  + Purple : right-mouth
  + Green : nose
  + Blue : left-eye
  + Orange : right-eye


(https://plot.ly/~stellastra666/156/)   
(https://plot.ly/~stellastra666/162/)

![이미지1](https://puu.sh/x6mIs/8cb7ee71c9.png) 
![이미지1](https://puu.sh/x9uGm/12e5061271.png)




### (2) landmark detection on CelebA

1. good case (red: predict / green: GT)

![이미지](https://puu.sh/x9DXw/b8a08644a8.png)
![이미지](https://puu.sh/x9DXJ/87499c145c.png)
![이미지](https://puu.sh/x9DXR/7e827d0363.png)
![이미지](https://puu.sh/x9DY2/a9db6c35fe.png)    
![이미지](https://puu.sh/x9DY6/860f44f4e7.png)
![이미지](https://puu.sh/x9DYe/f6d734eef3.png)
![이미지](https://puu.sh/x9DYl/ecae735bde.png)
![이미지](https://puu.sh/x9DYs/16f7b374c6.png)

2. badcase

![이미지](https://puu.sh/x9E9P/70052b1074.png)
![이미지](https://puu.sh/x9E9L/b79143d3ee.png)
![이미지](https://puu.sh/x9E9B/e97da93f86.png)
![이미지](https://puu.sh/x9E9t/02cb77ea76.png)    
![이미지](https://puu.sh/x9E9o/7cc9298d1c.png)
![이미지](https://puu.sh/x9E9i/aa01b2db50.png)
![이미지](https://puu.sh/x9E9a/eb7b5d3f29.png)
![이미지](https://puu.sh/x9E91/d765dec3a1.png)

### (3) Performance Benchmark

__1. Original paper__

|  nLandmark |  regressor training  | IOD error|
| ---- | --- |---|
|10|CelebA|6.32|
|30|CelebA|5.76|
|50|CelebA|5.33|

__2. My code__

|  nLandmark |  regressor training |Iter(reg) | MSE  | IOD error|
| ---- | --- |---|---|---|
|100|CelebA|5K|      3.15|5.71|
|100|CelebA|50K|      3.31|5.67|

### (4) Effect of training data when fine-tuning regressor(mlp)

|Training images| learning iter | training loss | MSE | IOD error|
|---|---|---|---|---|
| 10 | 1K | 0.04 |5.67  | 9.97  |
| 50 | 1K | 0.09 |4.73  | 8.07  |
| 100 | 1K | 0.13 |4.42  | 8.13  |
| 2000 | 2K | 0.18 |3.38  | 6.28  |
| 5000 | 3K | 0.20 |3.36  | 5.84  |
| 15000 | 5K | 0.21 |3.15  | 5.71  |
| 15000 | 50K | 0.21 |3.31  | 5.67  |

# ACKNOWLEDGEMENT
Thank James for kindly answering my inquries and providing pieces of matlab code :)


# Author
MinchulShin / [@nashory](https://github.com/nashory)


