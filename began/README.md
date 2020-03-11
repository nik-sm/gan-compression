Searching for a center crop:

```
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import transforms

path = '/data/niklas/datasets/celeba/img_align_celeba'

imgs = [
'012870.jpg',
'046637.jpg',
'080404.jpg',
'114171.jpg',
'147938.jpg',
'181705.jpg']

imgs = [f'0000{str(x).zfill(2)}.jpg' for x in range(10,20)]
print(imgs)
#left_right = (218 - 128)

# tmp = lambda img:


t1 = transforms.Compose([
#    transforms.Lambda(lambda img: transforms.functional.crop(img, 40,15,150, 150)),
    transforms.Lambda(lambda img: transforms.functional.crop(img, 51,26,128, 128)),

#     transforms.CenterCrop((150,128)),
#     transforms.Resize((128,128)),
    transforms.ToTensor()])


t2= transforms.Compose([
    transforms.Lambda(lambda img: transforms.functional.crop(img, 40,15,150, 150)),
#    transforms.Lambda(lambda img: transforms.functional.crop(img, 51,26,128, 128)),

#     transforms.CenterCrop((150,128)),
     transforms.Resize((128,128)),
    transforms.ToTensor()])



fig, ax = plt.subplots(3,len(imgs), figsize=(16,9))

for index, img in enumerate(imgs):
    img = Image.open(os.path.join(path, img))
    ax[0][index].imshow(img)
    transformed = t1(img)
    ax[1][index].imshow(transformed.permute(1,2,0))
    transformed2 = t2(img)
    ax[2][index].imshow(transformed2.permute(1,2,0))

    print(transformed.shape)
    print(transformed2.shape)
plt.show()
```
