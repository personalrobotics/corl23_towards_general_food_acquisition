# Towards General Single-Utensil Food Acquisition with Human-Informed Actions

This repository, released under the BSD-3 license, contains code that accompanies the CoRL 2023 paper "[Towards General Single-Utensil Food Acquisition with Human-Informed Actions](https://openreview.net/forum?id=UZpWSDA3tZJ)". To cite it, please cite the paper:

> Gordon, E. K., Nanavati, A., Challa, R., Zhu, B. H., Faulkner, T. A. K., & Srinivasa, S. (2023, August). Towards General Single-Utensil Food Acquisition with Human-Informed Actions. In 2023 Conference on Robot Learning (CoRL 2023).

## Dependencies
- Python (3.11.3)
- [OpenCV (4.7.0)](https://pypi.org/project/opencv-python/)
- [matplotlib (3.7.1)](https://matplotlib.org/stable/users/installing/index.html)
- [numpy (1.24.2)](https://numpy.org/install/)
- [scipy (1.10.1)](https://scipy.org/install/)
- [transformations (2022.9.26)](https://pypi.org/project/transformations/)

## Usage

### Extracting Actions from Human Data

#### Step 1: Get Data

Download one or more trials from [the dataset published alongside the paper](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/C8SI1D) and store them in the same folder. The extraction script expects a structure like this, which will be created by default when unzipping the files from the [published dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/C8SI1D):
> your_raw_data_folder/
>     camera_instrinsics.csv
>     subject{subject_num}_{food_name}/
>         {trial_num}/
>             {timestamp}_depth.png
>             {timestamp}_rgb.jpg
>         {trial_num}_static_transforms.csv
>         {trial_num}_wrenches_poses.csv

Note that in theory you can collect your own data of humans acquiring food items, as long as it is in the same structure as the dataset (the data structure is documented in the [README of the dataset](https://dataverse.harvard.edu/file.xhtml?fileId=6690561&version=1.0)). However, some aspects of `script/extract_actions.py` are particular to our data (e.g., trials to remove, CV that is tuned to a blue plate, etc.), so that file will have to be generalized somewhat to accomodate another dataset. For greatest reliability when collecting your own data, we recommend mimicing the data collection setup presented in the paper as closely as possible.

#### Step 2: Extracting Actions

Run `python3 scripts/extract_actions.py "/path/to/your_raw_data_folder"`. This should generate `data/action_schema_data.csv`. To gain more visibility into the extraction process, we recommend passing in a relative path to the optional `--output_img_path` parameter (e.g., `output_img_path data/img_outputs`). The script also has other parameters, which can be found using `python3 scripts/extract_actions.py -h`.

Note that even if you use the entire dataset, there will be small differences between `data/action_schema_data.csv` and the actions we extracted for the paper, `data_original/action_schema_data.csv`. This is because the action schema data used in the paper was extracted from the raw data (rosbags), not the published dataset. There are slight changes in computations between the two (e.g., the rosbags had rotations as quaternions whereas the published dataset has rotations as XYZ euler angles) that can accumulate floating-point errors. Further, there were slightly different inclusion criteria across the datasets -- e.g., the published dataset rejected a trial if there was a tracking error anytime in the trajectory, including at the end, whereas the script used for the paper ignored tracking errors after the fork reached the mouth. For those reasons, we also include the extracted actions used in the paper in `data_original/action_schema_data.csv`, so folks can recreate the clustering analysis with the actions from the paper.

### Clustering Actions (K-Medoids)

#### Step 1: Find the Elbow Point

Run `python3 scripts/cluster_actions.py` to run clustering on `data/action_schema_data.csv`. This should generate `data/k_medoids_by_k.png` which visualizes how the clustering quality changes with k, and shows the elbow point.

**NOTE**: To change the folder it gets input and output from, run `python3 scripts/cluster_actions.py --data_path /path/to/folder`. This can be particularly useful if you want to run clustering on the original extracted actions as opposed to extracting your own actions: `python3 scripts/cluster_actions.py --data_path data_originalaction_schema_data.csv`

#### Step 2: Get the Medoids

Run `python3 scripts/cluster_actions.py --k {num}` to get the medoids for the specified value of k. For example, to get the medoids used in the original paper, run `python3 scripts/cluster_actions.py --data_path data_original/action_schema_data.csv --k 11`. This should output a file, e.g., `data/kmedoids_centers_k_11.csv` with the representative (medoid) actions per cluster.

### Executing an Action on the Robot

Executing a point in the action schema on a robot is robot-specific. To see how we implemented it on the Kinova JACO 6-DOF arm for the experiments in the paper, see [here](https://github.com/personalrobotics/ada_feeding/blob/egordon/posthoc/trees/feeding/acquisition.xml). Note that some manual tweaking of the actions may be necessary to get them to execute on the robot hardware. For the paper, the manual tweaks we did were: (1) setting small angular twists (<5 deg) to 0; and (2) setting all grasp torque thresholds to 1.0 and all other torque thresholds to 4.0, to account for sensitivity issues in our force-torque sensor.