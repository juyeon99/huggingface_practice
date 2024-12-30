# image tag to text: https://github.com/xinyu1205/recognize-anything/blob/main/inference_tag2text.py

import argparse
import numpy as np
import random
import torch
from PIL import Image
from ram.models import tag2text
from ram import inference_tag2text as inference
from ram import get_transform

# parser = argparse.ArgumentParser(description='Tag2Text inferece for tagging and captioning')
# parser.add_argument('--image',
#                     metavar='DIR',
#                     help='path to dataset',
#                     default='images/1641173_2291260800.jpg')
# parser.add_argument('--pretrained',
#                     metavar='DIR',
#                     help='path to pretrained model',
#                     default='pretrained/tag2text_swin_14m.pth')
# parser.add_argument('--thre',
#                     default=0.68,
#                     type=float,
#                     metavar='N',
#                     help='threshold value')
# parser.add_argument('--specified-tags',
#                     default='None',
#                     help='User input specified tags')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = get_transform(image_size=384)

# delete some tags that may disturb captioning
# 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]

#######load model
model = tag2text(pretrained='tag2text_swin_14m.pth',
                            image_size=384,
                            vit='swin_b',
                            delete_tag_index=delete_tag_index)
model.threshold = 0.68  # threshold for tagging
model.eval()

model = model.to(device)

image = transform(Image.open('1641173_2291260800.jpg')).unsqueeze(0).to(device)

res = inference(image, model, '')
print("Model Identified Tags: ", res[0])
print("User Specified Tags: ", res[1])
print("Image Caption: ", res[2])

### Result:
# Model Identified Tags:  trail | flower | path | mountain | road | hill | lake | yellow
# User Specified Tags:  None
# Image Caption:  a winding road with yellow flowers on the side and a lake and mountains in the distance