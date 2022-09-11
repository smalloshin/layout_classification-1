# layout_classification

## Installation With Docker

```=sh
docker build -t streamlit .
docker run -p 8501:8501 streamlit
```
> Based on your server's network configuration, you could map to port 80/443 so that users can view your app using the server IP or hostname. For example: http://your-server-ip:80 or http://your-hostname:443.

## Installation -without docker

```=sh
conda create -n layout python=3.9
pip install streamlit
conda install -c pytorch torchvision torchaudio pytorch
brew install ffmpeg pkg-config
conda install jpeg libpng
pip install opencv-python
conda install grpcio
conda install pycocotools
pip install git+https://github.com/facebookresearch/detectron2@main
pip install av
```

## Project Structure

    .
    ├── app
    │   ├── main.py
    │   ├── visualize.py
    │   ├── detect.py
    │
    ├── samples
    │   ├── *.png
    │

- main.py: main entry point of prototype
- detect.py: functions and setup of layout classification modelling
- visualize.py: functions to visualize layout outputs in the prototype

## Prototype Usage

After select the input image or upload the PDF file, one can choose a model based on the content of the images.

- Choices of Model:
    - Magazine: Trained from the PrimalLayout dataset, favors magazine
    images, and includes 6 different classes: {1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"}
    - NewsPaper: Trained from the NewspaperNavigator dataset, it contains 7 classes of layout mainly targeting newspapers,
    {0: "Photograph", 1: "Illustration", 2: "Map", 3: "Comics/Cartoon", 4: "Editorial Cartoon", 5: "Headline", 6: "Advertisement"}
    - Academic Papers: Trained from the PubLayNet dataset, mostly used in academic paper and reports
    , it includes 5 classes:{0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}
- Choice of HyperParameters:
Choose detection Score threshold and NMS threshold can effectively remove false positives or increase accuracy, use with care. 
 ![image](https://user-images.githubusercontent.com/358454/184547439-8c84735e-1293-4400-bd73-88109332b594.png)
    
