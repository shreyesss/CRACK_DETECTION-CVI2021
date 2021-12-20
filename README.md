
# Pix2Pix

Pix2pix is a U-net GAN implemented from the imagetoimage translation paper CVPR. Image-to-image translation
as the task of translating one possible representation of a
scene into another, given sufficient training data.  [Link to paper](https://arxiv.org/pdf/1611.07004.pdf) 

## Architecture
The Generator is made up of an encoder-decoder architecture with skip connections between the corresponding layers of encoder and decoder. To give the generator a means to circumvent the bottleneck for information like this, skip connections are added , following the general shape of a “U-Net” . Specifically, the
skip connections are  between each layer i and layer n − i,
where n is the total number of layers. Each skip connection simply concatenates all channels at layer i with those
at layer n − i.


![image](https://user-images.githubusercontent.com/84932711/146767550-9e37105d-4008-421b-8e5c-f76b773dbcff.png)


The Discriminator is 30x30 PatchGAN. . This is advantageous because a smaller PatchGAN has fewer parameters, runs faster, and can be applied to arbitrarily large images. 
Both generator and discriminator use modules of the form convolution-BatchNorm-ReLu. 


![image](https://user-images.githubusercontent.com/84932711/146768014-29bb4788-25ed-4cb9-b363-e09f496701e6.png)




## Loss Functions

The objective Function is


G∗ = arg minGmaxDLcGAN (G, D) + λLL1(G) 

where G tries to minimize this objective against an adversarial D that tries to maximize it, i.e. G∗ =
arg minG maxD LcGAN (G, D).

The model uses a weighted mean of L1 and cross entropy loss as the loss function.  

## Data
pix2pix google drive link: https://drive.google.com/drive/folders/118NGnXl2K7tdcfHJeWTNEDeZvw5SuFK_?usp=sharing
Make a copy of this in your drive
