# GANs for Lossy Image Compression

We perform lossy image compression and compressive sensing using a BEGAN model trained on CelebA.

We show how to smoothly vary the latent dimension of the model without retraining.
Some of our compressive sensing results are [bananas!](#compressive-sensing-on-bananas)

For more details, see our [writeup](gans-for-lossy-image-compression.pdf).

# Usage

## Data Preprocessing
See `preprocess_images.py`.

```bash
python preprocess_images.py --dataset celeba --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_DIR
```

## Training
See `train.py`.

```bash
python train.py --run_name $RUN_NAME \
                --dataset celeba \
                --latent_dim 64 \
                --dataset_dir $OUTPUT_DIR \
                --epochs 50 \
                --n_train -1 \
                --output_activ elu
```

## Compression
The file `compress.py` offers both a `compress()` and an `uncompress()` function.
See the docstring for an explanation of the compressed file format.

```python
input_filename = 'bananas.jpg'
compressed_filename = 'bananas.ganz'
reconstr_filename = 'bananas_reconstr.png'

# Load the image
torch_img = load_target_image(input_filename)

# Compress
x_hat, z, psnr_gan, file_size = compress(torch_img,
                                         output_filename=compressed_filename,
                                         skip_linear_layer=True,
                                         no_linear_layer=False,
                                         compression_ratio=6,
                                         compressive_sensing=False,
                                         n_steps=n_steps)

# Uncompress
x_hat = uncompress(compressed_filename, reconstr_filename)
```

# Compressive Sensing on Bananas
After training on faces, our method achieves a PSNR of 24.76dB on a photo of bananas!
(here we use M=4900 Gaussian measurements on an image of size 128x128x3, or 
approximately 10% measurement ratio)
![High quality compressive sensing on an out-of-domain image](figures/bananas.comp.CR%3D6.CS%3DTrue.n_steps%3D7500.n_measure%3D4900.png)
