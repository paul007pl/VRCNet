# Point Cloud Completion Benchmark

### Introduction
An integrated Point Cloud Completion Benchmark implemented in Python 3.5, PyTorch 1.2 and CUDA 10.0. Supported algorithms: PCN, Topnet, MSN, Cascade, ECG, VRC.

### Installation
1. Install dependencies:
+ h5py 2.10.0
+ matplotlib 3.0.3
+ munch 2.5.0
+ open3d 0.9.0
+ PyTorch 1.2.0
+ PyYAML 5.3.1

2. Download corresponding dataset (e.g. ShapeNet dataset, TopNet dataset or Cascade dataset)
3. Compile PyTorch 3rd-party modules ([ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch), [emd, expansion_penalty, MDS](https://github.com/Colin97/MSN-Point-Cloud-Completion), [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch))


### Usage
+ To train a model: run `python train.py -c *.yaml`, e.g. `python train.py -c pcn.yaml`
+ To test a model: run `python test.py -c *.yaml`, e.g. `python test.py -c pcn.yaml`
+ Config for each algorithm can be found in `cfgs/`.
+ `run_train.sh` and `run_test.sh` are provided for SLURM users. 

## Acknowledgement
We include the following PyTorch 3rd-party libraries:  
[1] [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)  
[2] [emd, expansion_penalty, MDS](https://github.com/Colin97/MSN-Point-Cloud-Completion)  
[3] [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)  

We include the following algorithms:  
[1] [PCN](https://github.com/wentaoyuan/pcn)  
[2] [MSN](https://github.com/Colin97/MSN-Point-Cloud-Completion)  
[3] [Topnet](https://github.com/lynetcha/completion3d)  
[4] [Cascade](https://github.com/xiaogangw/cascaded-point-completion)  
[5] [ECG](https://github.com/paul007pl/ECG)  
[6] VRC  


### License
Our code is released under MIT License.