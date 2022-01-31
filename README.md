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

| Before Training   | After 100 Epoch |
| ----------- | ----------- |
|![Image](Begin.png)   |   ![Image](After100.png)   |

### English -> German Europarl dataset
```
python train_translate.py
```

Training on a small subset of 1000 sentences (Included in this repo)
![Image](Loss.png)



