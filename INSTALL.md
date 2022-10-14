### Set up the python environment

```
conda create -n cylindgcn python=3.7
conda activate cylindgcn

# make sure that the pytorch cuda is consistent with the system cuda
# e.g., if your system cuda is 9.0, install torch 1.1 built from cuda 9.0
pip install torch==1.1.0 -f https://download.pytorch.org/whl/cu90/stable

pip install Cython==0.28.2
pip install -r requirements.txt

# install apex
cd
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 39e153a3159724432257a8fc118807b359f4d1c8
export CUDA_HOME="/usr/local/cuda-9.0"
python setup.py install --cuda_ext --cpp_ext
```

### Compile cuda extensions under `lib/csrc`

```
ROOT=/path/to/snake
cd $ROOT/lib/csrc
export CUDA_HOME="/usr/local/cuda-9.0"
cd ../extreme_utils
python setup.py build_ext --inplace
```

### Set up datasets

#### PAT datasets

1. Organize the dataset as the following structure:
    ```
    ├── /path/to/PAT
    │   ├── train
    │   │   ├── JPGEImages
    │   │   ├── instances_val.json 
    │   ├── val
    │   │   ├── JPGEImages
    │   │   ├── annotations.json 
    ```
2. Create a soft link:
    ```
    ROOT=/path/to/snake
    cd $ROOT/data
    ln -s /path/to/PAT PAT
    ```

#### OCT datasets

1. Organize the dataset as the following structure:
    ```
    ├── /path/to/OCT
    │   ├── train
    │   │   ├── JPGEImages
    │   │   ├── instances_val.json 
    │   ├── val
    │   │   ├── JPGEImages
    │   │   ├── annotations.json 
    ```
2. Create a soft link:
    ```
    ROOT=/path/to/snake
    cd $ROOT/data
    ln -s /path/to/OCT OCT
    ```
