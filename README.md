# PaintNet: 3D Learning of Pose Paths Generators for Robotic Spray Painting
##### Gabriele Tiboni, Raffaello Camoriano, Tatiana Tommasi.

This repository contains the code for the preprint [paper](https://arxiv.org/abs/2211.06930): "PaintNet: 3D Learning of Pose Paths Generators for Robotic Spray Painting".

*Abstract:* Optimization and planning methods for tasks involving 3D objects often rely on prior knowledge and ad-hoc heuristics. In this work, we target learning-based long-horizon path generation by leveraging recent advances in 3D deep learning. We present PaintNet, the first dataset for learning robotic spray painting of free-form 3D objects. PaintNet includes more than 800 object meshes and the associated painting strokes collected in a real industrial setting. We then introduce a novel 3D deep learning method to tackle this task and operate on unstructured input spaces—point clouds—and mix-structured output spaces—unordered sets of painting strokes. Our extensive experimental analysis demonstrates the capabilities of our method to predict smooth output strokes that cover up to 95% of previously unseen object surfaces, with respect to ground-truth paint coverage. [Watch video](https://gabrieletiboni.github.io/paintnet/)


<img src="https://www.gabrieletiboni.com/assets/spray_paint_task_outline_white.png" style="width: 80%; max-width: 900px; max-height: 320px;" />


Our release is **under construction**, you can track its progress below:

- [ ] PaintNet dataset for public download
- [ ] Code implementation
	- [x] dataset loader
	- [x] training
	- [x] PCD evaluation metric
	- [ ] results rendering
	- [ ] intra-stroke alignment
- [ ] Trained models


## Installation
This repo is not meant to be used as a python package. Just clone it, modify it and use it as you wish.
```
# Download PaintNet dataset at https://gabrieletiboni.github.io/paintnet/
# export PAINTNET_ROOT=<root/to/dataset/>

# git clone <this repo>
cd paintnet
pip install -r requirements.txt

cd torch-nndistance
python build.py install
```


## Getting Started
Run the `train.py` script to train a model and evaluate its performance on the test set by means of Pose-wise chamfer distance.

- *Complete (cuboids)*
	```
	python train.py --dataset cuboids-v1 \
			 --pc_points 5120 \
			 --traj_points 2000 \
			 --loss chamfer rich_attraction_chamfer \
			 --weight_rich_attraction_chamfer 0.5 \
			 --lambda_points 4 \
			 --extra_data orientnorm \
			 --weight_orient 0.25
			 --epochs 1250 \
			 --batch_size 32 \
			 --backbone pointnet2 --pretrained \
			 --seed 42
	```
- *Reproduce paper results* 
    - `python train.py --config cuboids_stable_v1.json --seed 42`
    - `python train.py --config cuboids_lambda1_v1.json --seed 42`
    - `python train.py --config windows_stable_v1.json --seed 42`
    - `python train.py --config shelves_stable_v1.json --seed 42`
    - `python train.py --config containers_stable_v1.json --seed 42`


## Citing
If you use this repository, please consider citing
```
@misc{tiboni2022paintnet,
  title = {PaintNet: 3D Learning of Pose Paths Generators for Robotic Spray Painting},
  author = {Tiboni, Gabriele and Camoriano, Raffaello and Tommasi, Tatiana},
  doi = {10.48550/ARXIV.2211.06930},
  keywords = {Robotics (cs.RO), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  publisher = {arXiv},
  year = {2022},
  primaryClass={cs.RO}
}
```