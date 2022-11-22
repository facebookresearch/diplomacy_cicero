
# Postman: Tensor RPCs

## Installation


### devfair

Set up a fresh environment:

```shell
module purge

module load cuda/9.2
module load cudnn/v7.3-cuda.9.2
module load NCCL/2.2.13-1-cuda.9.2

module load anaconda3
```

Create a new Conda environment, and install PolyBeast's requirements:

```shell
conda create -n postman python=3.7.5
conda activate postman
pip install numpy torch
```

Compile PyTorch
[yourself](https://github.com/pytorch/pytorch#from-source) or install it from a package.

If you install from a package and use cuda from modules, you need to patch paths:
```
conda install --yes pytorch=1.4 -c pytorch
sed -i -e 's#/usr/local/cuda/lib64/libnvToolsExt.so#/public/apps/cuda/9.2/lib64/libnvToolsExt.so#g' $CONDA_PREFIX/lib/python3.?/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake
sed -i -e 's#/usr/local/cuda/lib64/libcudart.so#/public/apps/cuda/9.2/lib64/libcudart.so#g' $CONDA_PREFIX/lib/python3.?/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake
sed -i -e 's#/usr/local/cuda/lib64/libculibos.a#/public/apps/cuda/9.2/lib64/libculibos.a#g' $CONDA_PREFIX/lib/python3.?/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake
```


Compile the [nest library](https://github.com/fairinternal/nest):

```shell
pip install nest/
```

Compile and install postman:

```shell
pip install -e postman/
```

## Example

See the [example folder](example/) for an example of a simple server and client.

## Development

We use [black](https://github.com/psf/black) as a code formatter. To
enable black as a pre-commit:

```shell
pip install pre-commit
pre-commit install
```
