# sprite-pipeline

You download images to a folder

![image](https://github.com/user-attachments/assets/c9571d74-167c-444a-85c0-de09c9c78eb6)

⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️

`python group.py --style painting.png -H 100 downloaded/`

![image](https://github.com/user-attachments/assets/f78390a6-6737-4d33-bf24-3554721d5bdd)

⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️

You open PS, resize layers, fix perspective, patch.

⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️

`python spritesheet.py graphics.psd`

Congrats! You get packed spritesheet and UV coordinates in .txt

## Details

1. [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) to extract foreground
2. [RetinexNet](https://github.com/weichen582/RetinexNet) to fix lights
3. [nst_vgg19](https://github.com/alexanderbrodko/nst_vgg19) for Neural Style Transfer
4. [RealESRGAN_MtG](https://huggingface.co/rullaf/RealESRGAN_MtG) to improve quality
5. OpenCV to other filters and algorithms
