## inpainting
Image inpainting using GAN and/or partial convolution with progressive growing training scheme.

**Warning**: progressive growing scheme might be problematic and use more gpu memory. In my experiments, no improvement was observed.

### Dataset
In my experiments, I use CelebA-HQ datasets. To create it, see [tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans).

You can use `generate_holes.py` to create rectangle holes with multiple resolutions. For irregular holes, `generate_irregular_holes.py` is a way to create it(probably not a good way).

### Related papers
* [Partial conv](https://arxiv.org/abs/1804.07723)
* GAN
* [pix2pix](https://arxiv.org/abs/1611.07004)


### Problems
* For large rectangle holes(ratio > 0.5), the results might be unnatural.
* When using partial conv in decoder, the model failed, this is because the masked area is not filled and learned. But [the paper](https://arxiv.org/abs/1804.07723) uses partial conv in decoder. I have no idea how to get it right. 
* Progressive growing training scheme uses more gpu memory than I think. May be it can be optimized. May be even the implementation is problematic.
* When pix2pix_style=True, gradient of discriminator vanished. But why?
