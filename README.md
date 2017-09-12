# dense\_equivariant.torch
Torch7 implementation of "Unsupervised object learning from dense equivariant image labelling"



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
first, download CelebA dataset[here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8).
~~~
<data_path>
			|-- image 1
			|-- image 2
			|-- iamge 3 ...
~~~

To train the feature extractor(CNN):
~~~
1. change options in "script/opts.lua" and "data/gen\_tps.py"
2. do "th pretrain.lua"
>> pretrained model will saved in 'repo/pretrain/'
~~~

To train the regressor(mlp):
~~~
1. change options in "script/opts.lua" and "data/gen\_reg.lua"
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
will be updated soon.





# Author
MinchulShin / @nashory

