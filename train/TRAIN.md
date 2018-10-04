# [Sketch Simplification](https://esslab.jp/~ess/research/sketch/)

## Overview

Th is the training code used in the research papers:

```
   "Learning to Simplify: Fully Convolutional Networks for Rough Sketch Cleanup"
   Edgar Simo-Serra*, Satoshi Iizuka*, Kazuma Sasaki, Hiroshi Ishikawa (* equal contribution)
   ACM Transactions on Graphics (SIGGRAPH), 2016
```

and

```
   "Mastering Sketching: Adversarial Augmentation for Structured Prediction"
   Edgar Simo-Serra*, Satoshi Iizuka*, Hiroshi Ishikawa (* equal contribution)
   ACM Transactions on Graphics (TOG), 2018
```

See our [project page](https://esslab.jp/~ess/research/sketch_master/) for more detailed information.


## Dependencies

- [Torch7](http://torch.ch)
- [randomkit](https://github.com/deepmind/torch-randomkit)
- [lfs](https://keplerproject.github.io/luafilesystem/)

Please install Torch7 using the [official
documentation](http://torch.ch/docs/getting-started.html). After being set up,
you should be able to get the remaining dependencies with

```
  luarocks install randomkit
  luarocks install luafilesystem
```

We note that the training code requires cuda and cudnn to work, and should be
included if you follow the Torch7 installation guide.

## Code Overview

There are two training scripts:

- `train.lua`: Contains code for training using the weighted MSE loss proposed in the SIGGRAPH 2016 paper.
- `train_adv.lua`: Contains code for training using the adversarial augmentation loss proposed in the TOG 2018 paper.

Note that to use `train_adv.lua` you must use a model pre-trained with
`train.lua`. This includes the pre-trained models provided in this repository
if you do not wish to train everything from scratch.

## Weighted MSE Training

In order to train a model, first a dataset of rough sketch and line drawing
pairs has to be prepared. Due to copyright issues, the dataset is not provided.
Once a dataset is obtained, a csv file containing all the pairs has to be
created. The format is the following:

```
/path/to/rough_sketch1.png,/path/to/line_drawing1.png
/path/to/rough_sketch2.png,/path/to/line_drawing2.png
  ...
```

Once the dataset csv file is saved as `train.csv`, Weighted MSE training can
then be started by running the following command:

```
th train.lua
```

On the first run, it will create a lot of temporary weight files in `wcache/`.
Note that this will take a while. Afterwards, it will load the entire dataset
in memory and start training. Every 2500 iterations, it will save the model
weights to a file in `cache/`. The script will run until killed.

For more options, see `th train.lua --help`.

## Adversarial Augmentation Training

For the adversarial augmentation training, two additional datasets should be
prepared: one containing only line drawings, and one containing only rough
sketches. Like the paired dataset, a csv file for each of the new datasets
should be created, with one image per line, such as the following:

```
/path/to/image1.png
/path/to/image2.png
...
```

The line dataset csv file should be saved as `train_line.csv` and the rough
sketch dataset csv file should be saved as `train_rough.csv`.

The adversarial augmentation training is done in two stages: first the
discriminator is trained using a pretrained simplification model. Afterwards
both the simplification and discriminator model are trained jointly. This is
done automatically with the `--pretraindnet` parameter that defaults to 1000.

```
th train_adv.lua
```

For more options, see `th train_adv.lua --help`.

Checkpoints will be saved to the `cache_adv/` directory. The script will run
until killed.

## Notes

- This was developed and tested on various machines from late 2015 to end of 2016.
- Due to the stochastic nature of adversarial training, results will change between runs.
- The adversarial training approach will eventually collapse if left training too long.

## Citing

If you use these models please cite:

```
@Article{SimoSerraSIGGRAPH2016,
   author    = {Edgar Simo-Serra and Satoshi Iizuka and Kazuma Sasaki and Hiroshi Ishikawa},
   title     = {{Learning to Simplify: Fully Convolutional Networks for Rough Sketch Cleanup}},
   journal   = "ACM Transactions on Graphics (SIGGRAPH)",
   year      = 2016,
   volume    = 35,
   number    = 4,
}
```

and

```
@Article{SimoSerraTOG2018,
   author    = {Edgar Simo-Serra and Satoshi Iizuka and Hiroshi Ishikawa},
   title     = {{Mastering Sketching: Adversarial Augmentation for Structured Prediction}},
   journal   = "ACM Transactions on Graphics (TOG)",
   year      = 2018,
   volume    = 37,
   number    = 1,
}
```

## Acknowledgements

This work was partially supported by JST CREST Grant Number JPMJCR14D1 and JST ACT-I Grant Numbers JPMJPR16UD and JPMJPR16U3.

## License

This sketch simplification code is  freely available for free non-commercial
use, and may be redistributed under these conditions. Please, see the [license](/LICENSE)
for further details.
