## Black and White Image Colorization with GAN
This is the code base for my final project of the Computer Vision 2024-fall course at ZJU. The codes are built mainly upon two repositories, 
[Image Colorization tutorial](https://github.com/moein-shariatnia/Deep-Learning/tree/main/Image%20Colorization%20Tutorial) and [let_there_be_color](https://github.com/pauljcb/let_there_be_color).
I made some changes to the training process, dataset and hyperparameters.

## How to run the model
1. Install the denpendencies.

    pip install -r requirements.txt

2. Prepare the dataset.

I use the ImageNet to train my model, please put the downloaded ImageNet dataset in the root directory. The ImageNet dataset should have the following structure.

    imagenet
        |____train
        |       |____n00001
        |       |       |______n00001_01.jpeg
        |       |       |______n00001_02.jpeg
        |       |       |______...
        |       |____n00002
        |       |       |______n00002_01.jpeg
        |       |       |______n00002_02.jpeg
        |       |       |______...
        |       |____... 
        |____val
              |____n00001
                |       |______n00001_01.jpeg
                |       |______n00001_02.jpeg
                |       |______...
                |____n00002
                |       |______n00002_01.jpeg
                |       |______n00002_02.jpeg
                |       |______...
                |____... 

So that it can be correctly loaded with `torchvision.datasets.ImageFolder`.

3. Run the training.

I trained the model on 6 A100-40G-PCIE for 100 epochs, which roughly takes 10 hours. To train the model from scratch, you need to run the following commands.

    cd src
    python3 main.py

To change the hyperparameters of the training process, please take a look at the codes in `main.py` and `model.py`.

