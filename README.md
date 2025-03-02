# UVC YOLOX
## Environment
* **python 3.8**
* **cuda 11.2**
* **torch 1.8.0**
## Installation
* **Download program**
```
git clone https://github.com/RVL224/UVC_YOLOX.git
```  
* **Create by Docker**
```
cd UVC_YOLOX-master/docker
docker build -t ryolox .
docker run -it -d --gpus all --name [YourContainerName] -v /path/to/dataset:/data -v /path/to/program:/workspace/YOLOX ryolox /bin/bash
```
* **install env from souce.**
```
pip3 install -v -e .
cd cuda_op
pip3 install -v -e .
cd ../DOTA_devkit
pip3 install -v -e .
```
## Train
* Dataset prepare
* **MVTec screw**  
1.Download  [MVTec screw](https://www.mvtec.com/company/research/datasets/mvtec-screws) dataset     
2.data transformation
```
cd data_tramsform
python jason2voc_MVTEC.py
```
* **costomer dataset**  
1.Install [labelimg2](https://github.com/chinakook/labelImg2)  
2.data transformation  
```
cd data_tramsform
python xml2voc_costomer.py
```
* **Start Training**  
1.Open the config file in `exps/example/yolx_voc/yolox_MVTEC_voc_s.py`,and modify the `data_dir`.  
2.Input the config file for training,for example:  
```
python3 tools/train.py -b 16 -f /path/to/your/config/file -c /path/to/pretrainmodel
``` 


  
