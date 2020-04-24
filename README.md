# ML-rock-paper-scissors

Machine learning application for game Rock-Paper-Scissors

## What is it?

Main purpose is to make algorithm to classify picture of hand to certain class, which is rock, paper or scissors.
Alternatively application can be used for any set of pictures.  

## Used technologies:

```
$ Python
$ joblib
$ matplotlib
$ numpy
$ pandas
$ Pillow
$ scikit-learn
```

## Installation

It is best to use the python `virtualenv` tool to build locally:

```
$ git clone https://github.com/BartekStok/ML-rock-paper-scissors
$ cd 05_ml_rps_game
$ virtualenv -p python3 venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
```

## Usage

1. First go to `settings.py` and set path to pictures folders. Then set 
 names for labels. Folders must be named just like the labels to recognize.
Lastly choose size of picture.
2. Go to `preprocess.py` and run the program. Be aware that all pictures
in given paths will be resized to chosen size!
3.  

## License

This project is licensed under the MIT License 

- Copyright 2020 © Bartłomiej Stokłosa