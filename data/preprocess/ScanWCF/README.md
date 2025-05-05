# ScanWCF Dataset Preprocessing
ScanWCF is a dataset for the instance scene completion task based on room layouts from ScanNet.

The ScanWCF dataset contains:
- Instance segmented partial point cloud scans
- Scene background meshes
- Scene annotation files for placing ShapeNet meshes (GT object instances) into scenes
- Free/occluded space constraints used by our model

You can request access to the ScanWCF dataset by completing the following form: [ScanWCF Terms of Use](https://docs.google.com/forms/d/e/1FAIpQLSd9pLMJssTMFYDZkiRdM_rh_E0k2MOsCswyK-tWaN6fm-6Fwg/viewform?usp=dialog)

Due to licensing, we do not provide the watertight ShapeNet meshes in our dataset. 
The instructions below provide details on how to process the ShapeNet models 
that are used as ground truth object meshes within our scene scans.

Requirements:
- numpy
- trimesh
- mesh2sdf

To properly finish setting up our ScanWCF dataset, perform the following steps:
1. Place our ScanWCF dataset into the `data` directory. If done correctly, the directory structure should look like:
    ```
    - point_based_instance_completion
        - data
            - ScanWCF
                - json_files
                - scenes
                ...
    ```
2. Download [ShapeNetV2](https://shapenet.org/) dataset.
2. To generate the watertight meshes used in our scans run:
    ```bash
        python data/preprocess/ScanWCF/process_object_meshes.py --shapenet_dir [path to ShapeNetV2 dataset] --scanwcf_dir data/ScanWCF/
    ```

Following this process should generate all the relevant data needed to train our scene completion model and place it under the `data/ScanWCF/` directory.