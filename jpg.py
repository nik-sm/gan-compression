from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

def jpeg_compress(img, quality_layers, quality_mode='rates'):
    """
        quality_mode: 'rates' - compression ratio. 'dB' - SNR value in decibels
    """
    img = Image.open(img)
    outputIoStream = BytesIO()
    img.save(outputIoStream, "JPEG2000", quality_mode=quality_mode, quality_layers=quality_layers)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

i = 'images/bananas.jpg'
results = jpeg_compress(i, [30], 'dB')
results.save('test.jpg')
