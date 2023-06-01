## 特性：
- 使用开源模型库[segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)或自定义模型
- 在大尺寸遥感影像上进行切片推理和融合拼接  
- 处理大尺寸遥感影像栅格及矢量数据的常用工具（基于GDAL）
- 使用.yml文件编写配置文件
- 易于魔改

## 环境：
- torch>1.6
- torchsummay
- segmentation_models_pytorch  
    `pip install git+https://github.com/qubvel/segmentation_models.pytorch` (或者直接下载工程然后python setup.py install)
- albumentations
- gdal
- opencv
- matplotlib
- scipy
- scikit-learn
- scikit-image
- tqdm


## 试例-使用GID数据集进行训练：
1 - 访问 https://x-ytong.github.io/project/GID.html 下载数据并解压；  
2 - 使用 tools/make_gid5_256.py 对数据进行样本切片；  
3 - 处理后样本集为以下目录结构：
```shell
    ├── {dataset_root}
    │   ├── train
    │       ├── images
    │       ├── labels
    │   ├── val
    │       ├── images
    │       ├── labels
```
4 - 修改 sgfm_b3.gid5.yml 上的数据路径，开启训练:
```shell
    python train.py -c sgfm_b3.gid5
```
将会在configs目录中加载sgfm_b3.gid5.yml配置文件，并在runs目录内建立同名目录存放训练数据。


5 - 推理：
```shell
    python infer.py -c sgfm_b3.gid5 -input ../datasets/test_data -output ../output
```
## 注意事项：
图像读写用到opencv（训练时）和gdal，注意opencv读入是按BGR排序的；  
开源模型库支持的模型请参考 https://github.com/qubvel/segmentation_models.pytorch/blob/master/README.md ，通过修改配置文件network_params可调用不同模型；部署到新环境需要联网下载预训练权重；  
