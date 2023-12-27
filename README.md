# LittleAppleWebUI

A WebUI that integrates dataset acquisition, post process and PLoRA / LoRA training, aiming to simpler dataset operations and manual training.
<br><br>
Almost all functions are from DeepGHS' work, these pipeline things are awesome! However, this training interface has no queue capabilities now.<br>
But we can do full-auto training now, select your favorite preset and fill in waifu name, then press the button!<br>
We have both [HCP](https://github.com/IrisRainbowNeko/HCP-Diffusion) & [kohya-ss](https://github.com/kohya-ss/sd-scripts) frameworks supported!
<br>
I didn't know that there was already webui support for plora training, so I tried writing this and later added some processing functions.
<br><br>
If you are interested in full-auto pipeline or image classification, it's recommended to follow the [DeepGHS team](https://github.com/deepghs), 
there are something interesting there :)

## Preview
![waifuc](https://github.com/LittleApple-fp16/LittleAppleWebUI/blob/master/resource/preview1.svg)
![tagger](https://github.com/LittleApple-fp16/LittleAppleWebUI/blob/master/resource/preview2.svg)
![illust](https://github.com/LittleApple-fp16/LittleAppleWebUI/blob/master/resource/preview3.svg)
![save](https://github.com/LittleApple-fp16/LittleAppleWebUI/blob/master/resource/preview4.svg)


## Installation
* Python 3.10.6
* Dependencies will be installed on first start.
* If there are something broken, delete `venv`.
* Torch can be install automaticlly now, CUDA version 11.8 and 12.1 can be detected from device.
* If your CUDA is not supported, you can also install it manually.

Install Command
```shell
git clone https://github.com/LittleApple-fp16/LittleAppleWebUI.git
```
## Usage
Run `webui.bat` on Windows<br>
For Linux, run `webui.sh` instead.

## Other
* WARN: This is a lightweight toolbox, and most of the work will be done in RAM.
* I don't understand Python, and I can't guarantee too much functionality for this. However, if there are any issues with UI usage, feel free to let me know.
* By the way, if you want advanced plora training, please contact HCP-Diffusion developers for assistance.
