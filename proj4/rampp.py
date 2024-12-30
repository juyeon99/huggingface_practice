# https://github.com/xinyu1205/recognize-anything
# pip install git+https://github.com/xinyu1205/recognize-anything.git
# !wget ram_plus_swin_large_14m.pth https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth

# STEP 1: Import modules
import numpy as np
import torch
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

# STEP 2: Create inference object
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download Model: https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth
model_path = "ram_plus_swin_large_14m.pth"
model = ram_plus(pretrained=model_path,
                            image_size=384,
                            vit='swin_l')
model.eval()
model = model.to(device)

# STEP 3: Load data
image_path = "demo1.jpg"
transform = get_transform(image_size=384)
image = transform(Image.open(image_path)).unsqueeze(0).to(device)

# STEP 4: Inference
res = inference(image, model)

# STEP 5: Post-processing
print("Image Tags: ", res[0])
# print("图像标签: ", res[1])


### Result:
# Image Tags:  armchair | blanket | lamp | carpet | couch | dog | gray | green | hassock | home | lay | living room | picture frame | piy | living room | picture frame | pillow | plant | room | wall lamp | sit | wood floor