<div align="center">
  <img src="images/logo.png" alt="Algorithm icon">
  <h1 align="center">infer_yolo_world</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_yolo_world">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_yolo_world">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_yolo_world/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_yolo_world.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

YOLO-World is a state-of-the-art real-time object detection model that leverages the power of open-vocabulary learning to recognize and localize a wide range of objects in images. 

![illustration1](https://www.yoloworld.cc/images/vis_lvis.png)

![illustration](https://www.yoloworld.cc/images/user_vocab.png)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_yolo_world", auto_connect=True)

# Run on your image
wf.run_on(url="https://images.pexels.com/photos/745045/pexels-photo-745045.jpeg?cs=srgb&dl=pexels-helena-lopes-745045.jpg&fm=jpg&w=1280&h=869")

# Display your image
display(algo.get_image_with_graphics())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str) - default 'yolo_world_m': Name of the YOLO_WORLD pre-trained model. Other model available:
    - yolo_world_s
    - yolo_world_l
    - yolo_world_l_plus
- **conf_thres** (float) default '0.1': Box threshold for the prediction [0,1].
- **max_dets** (int) - default '100': The maximum number of bounding boxes that can be retained across all classes after NMS (Non-Maximum Suppression).  This parameter limits the total number of detections returned by the model, ensuring that only the most confident detections are retained.
- **cuda** (bool): If True, CUDA-based inference (GPU). If False, run on CPU.

If using a custom model:
- **model_weight_file** (str, *optional*): Path to model weights file .pth. 
- **config_file** (str, *optional*): Path to model config file .py. 

**Parameters** should be in **strings format**  when added to the dictionary.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_yolo_world", auto_connect=True)

algo.set_parameters({
        "model_name": "yolo_world_m",
        "prompt": "person, dog, cup",
        "max_dets": "100",
        "conf_thres": "0.07"
        })

# Run on your image
wf.run_on(url="https://images.pexels.com/photos/745045/pexels-photo-745045.jpeg?cs=srgb&dl=pexels-helena-lopes-745045.jpg&fm=jpg&w=1280&h=869")

# Display your image
display(algo.get_image_with_graphics())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_yolo_world", auto_connect=True)

# Run on your image  
wf.run_on(url="https://images.pexels.com/photos/745045/pexels-photo-745045.jpeg?cs=srgb&dl=pexels-helena-lopes-745045.jpg&fm=jpg&w=1280&h=869")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

