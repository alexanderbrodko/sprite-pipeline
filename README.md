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

![image](https://github.com/user-attachments/assets/8f7ed3f2-b918-4268-a123-a919f21e2485)
![image](https://github.com/user-attachments/assets/8a650d62-fa07-426c-a2fa-600af78e0420)

Congrats! You get packed spritesheet and UV coordinates in .txt

## Details

1. [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) to extract foreground
2. [RetinexNet](https://github.com/weichen582/RetinexNet) to fix lights
3. [nst_vgg19](https://github.com/alexanderbrodko/nst_vgg19) for Neural Style Transfer
4. [RealESRGAN_MtG](https://huggingface.co/rullaf/RealESRGAN_MtG) to improve quality
5. OpenCV to other filters and algorithms


## Requirements
`pip install numpy opencv-python torch torchvision pillow psd-tools rectpack modelscope realesrgan nst_vgg19 gdown`



