# LittleAppleWebUI

A WebUI that integrates dataset acquisition, processing functions and PLoRA training, aiming to 
facilitate more complex dataset operations and simpler manual training.
<br><br>
Almost all functions are from DeepGHS' work, these things are awesome! However, this training interface has not queue capabilities.<br>
But we can do full-auto training now, select your favorite preset and fill waifu name, then press the button!<br>
We have both HCP & kohya-ss script!
<br>
I didn't know that there was already webui support for plora training, so I tried writing this and later added some processing functions. <br>As you can see, I am trying to restore a manual pipeline process.
<br><br>
If you are interested in full-auto pipeline, it's recommended to follow the [DeepGHS team](https://github.com/deepghs), 
there are something interesting there :)

## Installation
* Python 3.10.6
* Dependencies will be installed on first start.
* If there are something broken, delete `venv`.
* Torch can be install automaticlly now, CUDA version 11.8 and 12.1 can be detected from device,
* If your CUDA is not supported, you can also install it manually.
```shell
git clone https://github.com/LittleApple-fp16/LittleAppleWebUI.git
```
## Usage
start webui.bat

## Preview
![waifuc](https://github.com/LittleApple-fp16/LittleAppleWebUI/blob/master/markdown_res/preview1.svg)
![imgutils](https://github.com/LittleApple-fp16/LittleAppleWebUI/blob/master/markdown_res/preview2.svg)
![tagger](https://github.com/LittleApple-fp16/LittleAppleWebUI/blob/master/markdown_res/preview3.svg)
![settings](https://github.com/LittleApple-fp16/LittleAppleWebUI/blob/master/markdown_res/preview4.svg)

## Other
* WARN: This is a lightweight toolbox, and most of the work will be done in RAM.
* I don't understand Python, and I can't guarantee too much functionality for this. However, if there are any issues with UI usage, feel free to let me know.
* By the way, if you want advanced plora training, try [HCP-Diffusion-webui](https://github.com/7eu7d7/HCP-Diffusion-webui).
