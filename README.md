# SelfColoringUsingGAN

## 组成

* data  数据集合
* DataProcessing 处理数据集的代码集合



## 运行说明

### DataProcessing


前提：

* 需要 `python3.x` 的环境
* 安装`pip` 
* 需要电脑[配置OpenCL环境](https://documen.tician.de/pyopencl/misc.html)

不想装的话
```sh
链接:https://pan.baidu.com/s/1YmTIjA8O7rVJoDgUF7WV3g  
密码:lmiy
```


一些依赖包，可以通过以下方式安装：



numpy科学计算库

```sh
pip3 install numpy
```

 pyopencl并行计算库

```sh
pip3 install pyopencl
```

Pillow图像处理库

```sh
pip3 install Pillow
```

tqdm 进度条工具

```sh
pip3 install tqdm
```

然后进入DataProcessing文件夹

```sh
cd DataProcessing
```

运行

```sh
python3 GeneralDataSet.py
```

然后大概这个样

```sh
➜  DataProcessing git:(master) ✗ python3 GeneralDataSet.py 
59%|███████████████████████                | 2759/4659 [25:00<13:45,  2.30it/s]
```



程序将会在根目录的 `data/expand_data/test` 和`data/expand_data/train`下生成 `*_X.png`等的图片

用了OpenCL还是慢出翔，佛了佛了