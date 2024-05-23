# How to use

## Training Set
content: 
Image
```
wget -c http://images.cocodataset.org/zips/train2014.zip
```
Video Frames
```
wget -c http://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip
```
style:
```
pip install -U kaggle==1.5.3
kaggle competitions download -c painter-by-numbers -f train.zip
```
## Pyinn Installation
```
pip install git+https://github.com/szagoruyko/pyinn.git@master
```
## Compile
```
cd ops
sh ./make.sh
```
## Training

```
python train.py --style_weight 10.0 --conv gatconv --patch_size 4
```
Here is the link for pre-models and Best trained-models  [Models](
https://drive.google.com/drive/folders/1fBE7VixfRGDCU5vJ1CXgjAtxAxmJMVSp?usp=drive_link)

## Testing image
```
python test.py --conv gatconv --patch_size 4 
```
## Test video
```
python testVid.py --video_name xxx --style_video_name xxx --style_dir xxx
```
## GPU Resources
Please kindly note that our code needs at least 30 GB GPU memory to run. We used RTX 3080 when we trained and tested
It took 3 days for Image Task Training,1 day for Video Stask Training

## Acknowledgements
Our code is based on the wonderful work of [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) and [DPT](https://github.com/CASIA-IVA-Lab/DPT). We deeply appreciate their great codes!
## Example
Coming Soon
