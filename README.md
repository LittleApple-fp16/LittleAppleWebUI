# LittleAppleWebUI

A WebUI that integrates dataset acquisition, processing functions and PLoRA training, aiming to 
facilitate more complex dataset operations and simpler manual training.
<br><br>
Almost all functions are from DeepGHS' work, these things are awesome! However, this training interface is not a full-auto 
training pipeline and no queue capabilities.
<br>
I didn't know that there was already webui support for plora training, so I tried writing this and later added some processing functions. <br>As you can see, I am trying to restore a manual pipeline process.
<br><br>
If you are interested in full-auto pipeline, it's recommended to follow the [DeepGHS team](https://github.com/deepghs), 
there are something interesting there :)
## Installation
* Python 3.10.6
* Dependencies will be installed on first start.
* If there are something broken, delete `venv`.
* It is recommended to change the torch version according to the cuda version before installation.
```shell
git clone [repo]
```
## Usage
soon

## Other
* I don't understand Python, and I can't guarantee too much functionality for this. However, if there are any issues with UI usage, you can let me know.
* By the way, if you want advanced plora training, try [HCP-Diffusion-webui](https://github.com/7eu7d7/HCP-Diffusion-webui).