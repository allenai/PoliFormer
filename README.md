# PoliFormer: Scaling On-Policy RL with Transformers Results in Masterful Navigators

[![PoliFormer](https://img.shields.io/badge/PoliFormer-project-ff69b4.svg)](https://poliformer.allen.ai/)
[![arXiv](https://img.shields.io/badge/arXiv-2406.20083-b31b1b.svg)](https://arxiv.org/abs/2406.20083)
[![Conference](https://img.shields.io/badge/CoRL-2024-4b44ce.svg)](https://www.robot-learning.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the code and data for the paper "PoliFormer: Scaling On-Policy RL with Transformers Results in Masterful Navigators".

## 🐍 Setting up the Python environment 🐍

### 🐳 Docker 🐳 [Recommended]

Please see the [README.md](docker/README.md) in the `docker` directory for instructions on how to build and run the docker image.

or use the pre-built image from Docker Hub:

```bash
docker pull khzeng777/spoc-rl:v2
```
then:
```bash
export CODE_PATH=/path/to/this/repo
export DATA_PATH=/path/to/data
export DOCKER_IMAGE=khzeng777/spoc-rl:v2
docker run \
    --gpus all \
    --device /dev/dri \
    --mount type=bind,source=${CODE_PATH},target=/root/poliformer \
    --mount type=bind,source=${DATA_PATH},target=/root/data \
    --shm-size 50G \
    -it ${DOCKER_IMAGE}:latest
```
and use the following conda environment:
```bash
conda activate spoc
```

### 🛠 Local installation 🛠 [Not recommended]

```bash
pip install -r requirements.txt
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+966bd7758586e05d18f6181f459c0e90ba318bec
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+864913fpt2.1.2cu121
cd DETIC_PATH && git clone https://github.com/facebookresearch/Detic.git --recurse-submodules && cd Detic && $PIP install -r requirements.txt && mkdir models && wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

## 📊 Data 📊

### 📥 Downloading the training data 📥

PoliFormer is trained using `fifteen` from [SPOC](https://spoc-robot.github.io/). The `fifteen` type has the agent navigating and fetching one of fifteen possible object types. To download the training data for the `fifteen` type, run the following command:  

```bash
python -m scripts.download_training_data --save_dir /your/local/save/dir --types fifteen
```

for example
```bash
python -m scripts.download_training_data --save_dir data --types fifteen
```

#### 📁 Dataset format 📁

Once you run the above command, you will have a directory structure that looks like this
```
/your/local/save/dir/<fifteen OR all>_type
    <TASK_TYPE>
        house_id_to_sub_house_id_train.json # This file contains a mapping that's needed for train data loading
        house_id_to_sub_house_id_val.json   # This file contains a mapping that's needed for val data loading
        train
            <HOUSEID>
                hdf5_sensors.hdf5 -- containing all the sensors that are not videos
                    <EPISODE_NUMBER>
                        <SENSOR_NAME>
                raw_navigation_camera__<EPISODE_NUMBER>.mp4
                raw_manipulation_camera__<EPISODE_NUMBER>.mp4
        val
            # As with train
```


The `hdf5_sensors.hdf5` contains the necessary information to train PoliFormer, including the house id, starting pose, and target object type/id.

For more information about the downloaded data, including trajectory videos and recorded sensors, please refer to [SPOC](https://spoc-robot.github.io/) documentation.

## 🏋 Training and Evaluation 🏋

In order to run training and evaluation you'll need:

1. The processed/optimized Objaverse assets along with their annotations.
2. The set of ProcTHOR-Objaverse houses you'd like to train/evaluate on.
3. For evaluation only, a trained model checkpoint.

Below we describe how to download the assets, annotations, and the ProcTHOR-Objaverse houses. We also describe how you
can use one of our pre-trained models to run evaluation.

### 💾 Downloading assets, annotations, and houses 💾

#### 📦 Downloading optimized Objaverse assets and annotations 📦

Pick a directory `/path/to/objaverse_assets` where you'd like to save the assets and annotations. Then run the following commands:

```bash
python -m objathor.dataset.download_annotations --version 2023_07_28 --path /path/to/objaverse_assets
python -m objathor.dataset.download_assets --version 2023_07_28 --path /path/to/objaverse_assets
```

These will create the directory structure:
```
/path/to/objaverse_assets
    2023_07_28
        annotations.json.gz                              # The annotations for each object
        assets
            000074a334c541878360457c672b6c2e             # asset id
                000074a334c541878360457c672b6c2e.pkl.gz
                albedo.jpg
                emission.jpg
                normal.jpg
                thor_metadata.json
            ... #  39663 more asset directories
```

#### 🏠 Downloading ProcTHOR-Objaverse houses 🏠

Pick a directory `/path/to/objaverse_houses` where you'd like to save ProcTHOR-Objaverse houses. Then run: 
```bash
python -m scripts.download_objaverse_houses --save_dir /path/to/objaverse_houses --subset val
```
to download the validation set of houses as `/path/to/objaverse_houses/val.jsonl.gz`.
You can also change `val` to `train` to download the training set of houses.

#### 🛣 Setting environment variables 🛣

Next you need to set the following environment variables:
```bash
export PYTHONPATH=/path/to/poliformer
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_houses
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets
export DETIC_REPO_PATH=/path/to/DETIC_PATH
```

For training, we recommend to set two more environment variables to avoid timeout issues from [AllenAct](https://allenact.org/):
```bash
export ALLENACT_DEBUG=True
export ALLENACT_DEBUG_VST_TIMEOUT=2000
```

### 🚀 Running training 🚀
```bash
python training/online/dinov2_vits_tsfm_rgb_augment_objectnav.py train --num_train_processes NUM_OF_TRAIN_PROCESSES --output_dir PATH_TO_RESULT --dataset_dir PATH_TO_DATASET
```

for example
```bash
python training/online/dinov2_vits_tsfm_rgb_augment_objectnav.py train --num_train_processes 32 --output_dir results --dataset_dir data/fifteen/ObjectNavType
```


### 🚀 Running evaluation with a pretrained model 🚀

Download pretrained ckpt:
```bash
python scripts/download_trained_ckpt.py --save_dir PATH_TO_SAVE_DIR
```

for example:
```bash
python scripts/download_trained_ckpt.py --save_dir ckpt
```

Run evaluation using text-nav model:
```bash
python training/online/online_eval.py --output_basedir PATH_TO_RESULT --num_workers NUM_WORKERS --ckpt_path ckpt/text_nav/model.ckpt --training_tag text-nav --house_set objaverse --gpu_devices 0 1 2 3 4 5 6 7
```

Run evaluation using pure box-nav model:
```bash
python training/online/online_eval.py --output_basedir PATH_TO_RESULT --num_workers NUM_WORKERS --ckpt_path ckpt/box_nav/model.ckpt --training_tag text-nav --house_set objaverse --gpu_devices 0 1 2 3 4 5 6 7 --input_sensors raw_navigation_camera nav_task_relevant_object_bbox nav_accurate_object_bbox --ignore_text_goal
```

Run evaluation using text+box-nav model:
```bash
python training/online/online_eval.py --output_basedir PATH_TO_RESULT --num_workers NUM_WORKERS --ckpt_path ckpt/text_box_nav/model.ckpt --training_tag text-nav --house_set objaverse --gpu_devices 0 1 2 3 4 5 6 7 --input_sensors raw_navigation_camera nav_task_relevant_object_bbox nav_accurate_object_bbox
```

## 📝 Cite us 📝

```bibtex
@article{zeng2024poliformer,        
    author    = {Zeng, Kuo-Hao and Zhang, Zichen and Ehsani, Kiana and Hendrix, Rose and Salvador, Jordi and Herrasti, Alvaro and Girshick, Ross and Kembhavi, Aniruddha and Weihs, Luca},
    title     = {PoliFormer: Scaling On-Policy RL with Transformers Results in Masterful Navigators},
    journal   = {CoRL},
    year      = {2024},
}
```
