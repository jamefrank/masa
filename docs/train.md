# MASA Training Instructions

This is the instruction for training MASA. You can train MASA with any raw images you collected and transform your detector into a multiple object tracker.
MASA training is consist of two steps: (1). using SAM to segment every object in the raw images (2). training MASA with those segments. 
We describe these two steps in detail below.

## Segment Every Object 

### Using SA-1B-500K images
[SA-1B datasets](https://ai.meta.com/datasets/segment-anything/) is a huge datasets contains raw images from diverse open world scenarios. In the paper, we use a subset of 500K images sampled from SA-1B dataset to train the default MASA tracker.

#### 1. Download the SA-1B dataset
You can download the SA-1B dataset from [here](https://ai.meta.com/datasets/segment-anything-downloads/). You can create a folder to store SAM's data, eg ```data/sam/```. Then, extract images into one folder, eg. ```data/sam/batch0/```.

#### 2. Generate Segments Using SAM 
Since the SA-1B has already provide exhaustive segmentation generated by SAM. We can use them directly. For other raw images, you can run SAM-H model to get the segments of every object automatically in the images and save the results. We give an example on COCO images below.

#### 3. Generate annotations in COCO format
(a). Download the 500K image name list from [here](https://drive.google.com/file/d/1mFJhEpQLfEmZq27W323u3wosBTSSnj3r/view?usp=sharing). Then, you can put it in the ```data/sam/sam_annotations/jsons/sa1b_coco_fmt_iminfo_500k.json```

(b). Run following script to convert the annotations into coco format.

```aiignore
python tools/format_conversion/convert_sa1b_to_coco.py --img_list data/sam/sam_annotations/jsons/sa1b_coco_fmt_iminfo_500k.json --input_directory data/sam/batch0 --output_folder data/sam/sam_annotations/jsons/
```
After running the script, yuo will get two json files in the ```data/sam/sam_annotations/jsons/``` folder. One is the annotations of the segments, the other is containing the bounding boxes.
The bounding boxes are extracted from the segments. The latter is much smaller than the former, so we use the latter to train MASA. However, some advanced argumentation techniques may require the mask annotations, such as copy and paste, so we provide both of them.

### Using any raw images in your customized domain
You can also use any customize raw images for training your tracker. We give an example below of using COCO images.

#### 1. Prepare the Raw images 
You can download the COCO dataset from [here](https://cocodataset.org/#download). You can create a folder to store SAM's data, eg ```data/coco/```. Then, extract images into one folder, eg. ```data/coco/images/```.
#### 2. Generate Segments using SAM, and generated annotations in COCO format
You can use SAM-H model to get the segments of every object automatically. Specifically, you can run [this script](https://github.com/facebookresearch/segment-anything/blob/main/scripts/amg.py) to generate the segments of the raw images.
You need to install SAM first and follow the instructions in the original repo. Then, you will get the annotations in the SAM format. To convert the SAM format into COCO format, you can use the above script (```tools/format_conversion/convert_sa1b_to_coco.py```) we provided.

## Training MASA with Segments

After generating the segments, you can train MASA with the segments. We provide the training script in the ```tools/train.py```. You can run the following command to train MASA.
Here is a multiple GPU training using Grounding-Dino as example:

1. Download the pre-trained grounding-dino weights from [here](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth). Then, put it in the ```saved_models/pretrain_weights/``` folder.

2. Run the following command to train MASA with 8 GPUs:
```aiignore
tools/dist_train.sh configs/masa-gdino/masa_gdino_train.py 8 --work-dir saved_models/masa_gdino/
```
