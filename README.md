# CylinGCN: Cylindrical Structures Segmentation in 3D Biomedical Optical Imaging by a contour-based Graph Convolutional Network

![city](assets/cylingcn.png)

> [CylinGCN: Cylindrical Structures Segmentation in 3D Biomedical Optical Imaging by a contour-based Graph Convolutional Network](https://arxiv.org/pdf/)  
> Zhichao Liang, Shuangyang Zhang, Zhijian Zhuang, Xipan Li, Wufan Chen, Li Qi 

Any questions or discussions are welcomed!

## Installation

Please see [INSTALL.md](INSTALL.md).

## Testing

### Testing on PAT datasets
Test:
    ```
     python test.py --cfg_file configs/cylingcn_pat.yaml
    ```

### Testing on OCT datasets

Test:
    ```
    python test.py --cfg_file configs/cylingcn_oct.yaml
    ```
    

If setup correctly, the output will look like

![vis_city](assets/test.png)



## Training

The training parameters can be found in [project_structure.md](project_structure.md).

### Training on PAT datasets

```
python train.py --cfg_file configs/cylingcn_pat.yaml
```

### Training on OCT datasets

```
python train.py --cfg_file configs/cylingcn_oct.yaml
```



## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{Liang2022,
  title={CylinGCN: Cylindrical Structures Segmentation in 3D Biomedical Optical Imaging by a contour-based Graph Convolutional Network},
  author={Zhichao Liang, Shuangyang Zhang, Zhijian Zhuang, Xipan Li, Wufan Chen, and Li Qi},
}
```
