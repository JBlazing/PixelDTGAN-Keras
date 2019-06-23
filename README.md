# PixelDTGAN-Keras

This is my current work for my Master Thesis. This a Keras implementation for Pixel-Level Domain Transfer which was proposed by [Donggeun Yoo](https://dgyoo.github.io/). 

The main objective is to try and segmenet pictures of roads in order to identify the road. Using the [KITTI Road and Lane Data Set](http://www.cvlibs.net/datasets/kitti/eval_road.php)

Right Now this is a work in progress. Here is my current TO DO.

* [x] Implement the Selection Layer which chooses a output of the Generative network, the assocciated image, or the non associated image.
* [x] Switch to use Tensorflow 2.0 Keras interface 
* [x] Figure out optimal way to train the models either 3 seperate models or 1 giant model (Went with 3 seperate models)
* [x] Finish Training Function
* [x] Excedding GPU memory will figure out how to get around this without having to go get new GPU (This was a issue with GAN output layer sizing)
* [x] Optimize for ram memory usage (Try and load data when needed)
* [x] Try and figure out why the last batch won't compute gradients (Can't modify the output from the GAN to do a uniform  sampling from either fake , associated or disassociated)
* [x] Test on LookBook dataset (Some Images produce good results while others do not produce anything close)
* [x] Modify and experiment with the KITTI Dataset 
* [ ] Write report and then do Dissertation
