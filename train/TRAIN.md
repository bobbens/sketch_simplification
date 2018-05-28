# [Sketch Simplification](http://hi.cs.waseda.ac.jp/~esimo/research/sketch/)

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

See our [project page](http://hi.cs.waseda.ac.jp/~esimo/research/sketch_master/) for more detailed information.

## License

```
  Copyright (C) <2017-2018> <Edgar Simo-Serra and Satoshi Iizuka>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.

  Edgar Simo-Serra, Waseda University
  esimo@aoni.waseda.jp, http://hi.cs.waseda.ac.jp/~esimo/  
  Satoshi Iizuka, Waseda University
  iizuka@aoni.waseda.jp, http://hi.cs.waseda.ac.jp/~iizuka/
```

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

TODO.

## Notes

- Models are in Torch7 format and loaded using the PyTorch legacy code.
- This was developed and tested on various machines from late 2015 to end of 2016.

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


