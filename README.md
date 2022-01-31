# Simple Transformer

An implementation of the "Attention is all you need" paper without extra bells and whistles,
or difficult syntax.

Note: The only extra thing added is Dropout regularization in some layers and option to use GPU.

### Install
```
python -m pip install -r requirements.txt
```

### Toy data
```
python train_toy_data.py
```

![Image](Extra_Scripts/toy_data.gif)

### English -> German Europarl dataset
```
python train_translate.py
```

Training on a small subset of 1000 sentences (Included in this repo)
![Image](Extra_Scripts/Loss.png)



