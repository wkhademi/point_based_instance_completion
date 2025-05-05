<div align="center">

# Point-based Instance Completion with Scene Constraints

[Wesley Khademi](https://wkhademi.github.io), [Li Fuxin](https://web.engr.oregonstate.edu/~lif/)  
Oregon State University

<a href="https://arxiv.org/abs/2504.05698"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://wkhademi.github.io/point_based_instance_completion/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>

https://github.com/user-attachments/assets/0bbbb3ed-9918-46b9-aa4e-91f247df5c4b

</div>

Abstract: *Recent point-based object completion methods have demonstrated the ability to accurately recover the missing geometry of partially observed objects. However, these approaches are not well-suited for completing objects within a scene as they do not consider known scene constraints (e.g., other observed surfaces) in their completions and further expect the partial input to be in a canonical coordinate system which does not hold for objects within scenes. While instance scene completion methods have been proposed for completing objects within a scene, they lag behind point-based object completion methods in terms of object completion quality and still do not consider known scene constraints during completion. To overcome these limitations, we propose a point cloud based instance completion model that can robustly complete objects at arbitrary scales and pose in the scene. To enable reasoning at the scene level, we introduce a sparse set of scene constraints represented as point clouds and integrate them into our completion model via a cross-attention mechanism. To evaluate the instance scene completion task on indoor scenes, we further build a new synthetic dataset called ScanWCF, which contains labeled partial scans as well as aligned ground truth scene completions that are watertight and collision free. Through several experiments, we demonstrate that our method achieves improved fidelity to partial scans, higher completion quality, and greater plausibility over existing state-of-the-art methods.*

# Installation
Please follow the installation directions below to match the configuration we use for training/testing.

The required dependencies can be installed via conda and pip by running:
```bash
# Create conda environment
conda env create -f environment.yml
conda activate pbic

# Install hydra for managing configs
pip install hydra-core --upgrade

# Install NKSR
pip install nksr -f https://nksr.huangjh.tech/whl/torch-2.0.0+cu118.html

# Install PointNet++ ops
pip install libs/pointnet2_ops_lib/.
```

# Datasets

## Watertight ShapeNetV2 Dataset
We do not provide a download of our preprocessed ShapeNetV2 meshes used for pre-training our object completion model. Instead we provide scripts and directions for preprocessing the ShapeNetV2 dataset to be used for pre-training, which can be found [here](data/preprocess/ShapeNet55/README.md).

## ScanWCF Dataset
ScanWCF is a dataset for the instance scene completion task based on room layouts from ScanNet.

The ScanWCF dataset contains:
- Instance segmented partial point cloud scans
- Scene background meshes
- Scene annotation files for placing ShapeNet meshes (GT object instances) into scenes
- Free/occluded space constraints used by our model

You can request access to the ScanWCF dataset by completing the following form: [ScanWCF Terms of Use](https://docs.google.com/forms/d/e/1FAIpQLSd9pLMJssTMFYDZkiRdM_rh_E0k2MOsCswyK-tWaN6fm-6Fwg/viewform?usp=dialog)

Upon gaining access and downloading the dataset, place the downloaded data under the `data` directory to be used for training/testing. If done correctly, the directory structure should look like:
```
- point_based_instance_completion
    - data
        - ScanWCF
            - json_files
            - scenes
            ...
```

Due to licensing, we do not provide the watertight ShapeNet meshes in our dataset. 
Please refer to [here](data/preprocess/ScanWCF/README.md) for instructions on processing ShapeNet meshes to be used in our dataset.

# Model Checkpoints
If you want to train your own version of the scene completion model, you can download the weights of our object completion model pre-trained on ShapeNet to initialize the scene completion model from:
- [Object Completion Model Checkpoint](https://oregonstate.box.com/s/6s7ohvsenm0ldnxkdsx39oy7l0i0dpjo)

If you just want to run evaluation, you can directly download the weights of our scene completion model trained on our ScanWCF dataset:
- [Scene Completion Model Checkpoint](https://oregonstate.box.com/s/clhtmucyipim3ej4j8m3jai7jac31o4m)

To use downloaded model checkpoints, place the downloaded experiment directory under the `experiments` directory. By default, our config files point to the experiments of the provided model checkpoints (e.g., see `experiment_name` in `configs/test.yaml` or `pretrained_path` in `configs/train.yaml`).

# Train
To pre-train our object completion model on the watertight ShapeNet meshes, run:
```bash
python main/runner.py --config-name pretrain
```

We initialize our scene completion model from our pre-trained object completion model. To do so, you can provide the path to a saved model checkpoint in the `pretrained_path` parameter of the `configs/train.yaml` file. The saved model checkpoint can be generated from pre-training the object completion model yourself using the command above or by downloading our provided [Object Completion Model Checkpoint](https://oregonstate.box.com/s/6s7ohvsenm0ldnxkdsx39oy7l0i0dpjo) and placing the experiment under the `experiments` directory. Then to train our scene completion model on our ScanWCF dataset, run:
```bash
python main/runner.py --config-name train
```

# Test
Mask3D training and testing is currently not implemented in this repo. In the meantime, we provide a download link to our Mask3D instance segmentation predictions that we used for evaluation: [Mask3D Predictions](https://oregonstate.box.com/s/azdki8ya3yeo3fosfbr2q5ty59sda36v). The downloaded directory can be placed directly under the `data` directory to be used for evaluating our scene completion model.

To run the scene completion model using the Mask3D instance predictions of partial scans use:
```bash
python main/runner.py --config-name test_mask3d
```

To run the scene completion model using the ground truth instance segmentations of partial scans use:
```bash
python main/runner.py --config-name test
```

# Visualization
We provide a script for visualizing predicted completions, predicted surface normals, and reconstructed meshes. To visualize predictions, run:
```bash
# visualize all scenes using Mask3D instance segmentations
python main/visualize.py --data_dir ./data/ScanWCF --pred_dir ./experiments/{experiment name}/results_Mask3D/completions

# visualize all scenes using ground truth instance segmentations
python main/visualize.py --data_dir ./data/ScanWCF --pred_dir ./experiments/{experiment name}/results_ScanWCF/completions

# visualize a specific scene
python main/visualize.py --data_dir ./data/ScanWCF --pred_dir ./experiments/{experiment name}/{results_Mask3D or results_ScanWCF}/completions --scene_id {scene id}_{partial id}
```

# Evaluation
Use the following commands to run instance scene completion metrics:
```bash
# chamfer distance (CD)
python evaluation/cd/eval.py --data_dir ./data/ScanWCF --pred_dir ./experiments/{experiment name}/results_Mask3D/reconstructions

# intersection of union (IoU)
python evaluation/iou/eval.py --data_dir ./data/ScanWCF --pred_dir ./experiments/{experiment name}/results_Mask3D/reconstructions

# light field distance (LFD)
python evaluation/lfd/eval.py --data_dir ./data/ScanWCF --pred_dir ./experiments/{experiment name}/results_Mask3D/reconstructions

# Point Coverage Ratio (PCR)
python evaluation/pcr/eval.py --data_dir ./data/ScanWCF --pred_dir ./experiments/{experiment name}/results_Mask3D/reconstructions
```

Use the following command to run scene completion metrics (i.e., completion results when ground truth instance segmentation is provided):
```bash
python evaluation/scene_completion_evaluation.py --data_dir ./data/ScanWCF --pred_dir ./experiments/{experiment name}/results_ScanWCF/completions
```

# Citation
```
@inproceedings{khademipoint,
  title={Point-based Instance Completion with Scene Constraints},
  author={Khademi, Wesley and Li, Fuxin},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

# Acknowledgements
Some parts of the code are borrowed from other works. We thank the authors for their work:
- [NKSR](https://github.com/nv-tlabs/NKSR)
- [Mask3D](https://github.com/JonasSchult/Mask3D)
- [SeedFormer](https://github.com/hrzhou2/seedformer)
- [DIMR](https://github.com/ashawkey/dimr)
- [PCN](https://github.com/wentaoyuan/pcn)
