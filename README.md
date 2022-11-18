## Improving Unsupervised Label Propagation for Pose Tracking and Video Object Segmentation
This repository provides code for [Improving Unsupervised Label Propagation for Pose Tracking and Video Object Segmentation](https://urs-waldmann.github.io/improving-unsupervised-label-propagation/) (GCPR 2022).

**Abstract**

Label propagation is a challenging task in computer vision with many applications. One approach is to learn representations of visual correspondence. In this paper, we study recent works on label propagation based on correspondence, carefully evaluate the effect of various aspects of their implementation, and improve upon various details. Our pipeline assembled from these best practices outperforms the previous state of the art in terms of PCK_0.1 on the JHMDB dataset by 6.5%. We also propose a novel joint framework for tracking and keypoint propagation, which in contrast to the base pipeline is applicable to tracking small objects and obtains results that substantially exceed the performance of the core pipeline. Finally, for VOS, we extend our pipeline to a fully unsupervised one by initializing the first frame with the self-attention layer from DINO. Our pipeline for VOS runs online and can handle static objects. It outperforms unsupervised frameworks with these characteristics.

If you find a bug, have a question or know how to improve the code, please open an issue.

## Setup

The main sources are located in `src/`.

### Python requirements

The PyTorch builds with CUDA aren't available in PyPi and need to be installed manually first:

```sh
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Then, install the requirements listed in `./requirements.txt`, e.g.

```sh
python -m pip install -r requirements.txt
```

If you don't have a CUDA-ready device installed use the `./requirements-cpu.txt`. The code should fall back to a
supported implementation of necessary algorithms.

### Setting the python path

All further instructions assume, that the `PYTHONPATH` variable was set to point to the `./src` directory (e.g. in bash
`export PYTHONPATH="./src"` or in Powershell `$env:PYTHONPATH='./src'`).

### Datasets

The code is organized, such that paths are configured in `./config.py`. This file dynamically configures the path prefix
used by default. This allows to easily switch between different machines or operating systems. It is easiest, to set the
`WORKSPACE_PREFIX` variable to a directory where all processing should happen. Alternatively, you can also set some
paths manually to fit your needs.

The most important paths are:

```txt
<prefix>/                           # Workspace directory
         datasets/                  # Base for datasets
                  DAVIS/DAVIS2017/  # DAVIS2017 data
                  JHMDB/            # JHMDB data
                  SegTrackV2        # SegTrackV2 data
...                                 # More datasets
         results/                   # Location for results
         checkpoints/               # Albeit not used much. Models are cached in the usual PyTorch Hub caching location.
```

### Preparing the datasets

#### DAVIS

Download: <https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip> and extract the contents to
the configured dataset root, e.g. `<prefix>/datasets/DAVIS/DAVIS2017/` by default. This file contains everything
necessary for DAVIS2016 and DAVIS2017.

#### SegTrackV2

Download and extract <https://web.engr.oregonstate.edu/~lif/SegTrack2/SegTrackv2.zip>

#### JHMDB

Run the following tool to download and extract the necessary archives automatically.

```sh
python -m dataset.preparation.prepare_JHMDB --output <output-path> --extract
```

Add the `--download-all` to download not only the necessary files but also archives for body part masks etc.

## Usage

The code is built, such that most things can be done by simply creating a configuration and then running:

```shell
python -m tools.evaluate_config --config "<path-to-config>" --output-dir "<output-dir>"
```

The config path can be prefixed with `$config/` to load configurations from the `./share/config/` directory.

The most important configurations are located in the `$config/label_propagation/_reported_configs/` directory.

### Creating configurations

The configuration schema is defined in `./share/config/schema/config-schema.json`. Configurations are checked against
this schema prior to instantiation. The general structure of the configuration has four blocks, one for the feature
extractor, one for the dataset, one for the label propagation and one for the evaluation mode. An example of such a
configuration is shown below.

```json
{
  "feat_ext": {
    "name": "vit", "variant": "base", "patch_size": 8,
    "scale_size": 480
  },
  "dataset": {
    "name": "segmentation_dataset",
    "data": {
      "name": "davis", "year": "2017", "split": "val",
      "mode": "multi-object"
    },
    "codec": {
      "channel_normalization": "minmax",
      "interpolation_mode": "bicubic"
    }
  },
  "label_propagation": {
    "name": "affinity",
    "implementation": "local",
    "feature_normalization": true,
    "affinity_topk": 5,
    "topk_implementation": "full",
    "affinity_norm": "dino",
    "neighborhood_size": 12,
    "label_normalization": "none"
  },
  "evaluator": {
    "num_context": 7,
    "recreate_labels": false,
    "label_initializer": "ground_truth"
  }
}
```

This configuration defines multi-object O-VOS inference on DAVIS2017.

```json
{
  "feat_ext": {"name": "vit", "variant": "base", "patch_size": 8, "scale_size": 480},
  ...
}
```

This block defines the feature extractor as a vision transformer (DeIT) of size `base` and patch size `8`, i.e. `DeIT-B/8`. The
scale size indicates, how the input images should be resized. `480` implies, that the shorter side is rescaled to `480`
pixels. If no weight source is configured, as is the case here, weights trained with DINO are loaded, if they are
available.

```json
{
  ...
  "dataset": {
    "name": "segmentation_dataset",
    "data": {
      "name": "davis", "year": "2017", "split": "val",
      "mode": "multi-object"
    },
    "codec": {
      "channel_normalization": "minmax",
      "interpolation_mode": "bicubic"
    }
  },
  ...
}
```

This block defines that we want to use the validation split of the DAVIS2017 dataset. `multi-object` indicates, that the masks are
loaded as multi-object masks, i.e. indexed instead of binary. The `codec` object defines properties of the mask to label
translation, for example the scaling mode to adapt between label and mask size.

The `"label_propagation"` object that follows, defines key parameters of the label propagation implementation. Here, it
is advised to have a look at the available configuration options and the corresponding implementation.

Finally, `"evaluator": {"num_context": 7, "recreate_labels": false, "label_initializer": "ground_truth"}` defines our
inference. `num_context` is the number of context frames used during propagation. `recreate_labels` defines, whether the
labels should be recreated after every frame, i.e. performing a decoding and encoding step. Finally, the
`label_initializer` option defines if we are using O-VOS inference or Z-VOS by choosing an implementation for the
selection of the initial mask.

### Evaluating the results

#### Evaluating JHMDB

```shell
# JHMDB-test1: Evaluate
python -m dataset.evaluation.evaluate_JHMDB <result-dir> --compute-coverage
```

You can evaluate multiple runs at once by setting the `--eval-all` flag and passing a directory that contains a subdir
for each of the runs.

To evaluate a different variant of JHMDB that is not `test` split `1` add the `--dataset-split-num <num>` and
`--dataset-split-name <split-name>` arguments.

For more options see `python -m dataset.evaluation.evaluate_JHMDB --help`

#### Evaluating segmentation datasets

```shell
# DAVIS2017val: Evaluate multi-object
python -m dataset.evaluation.evaluate_segmentation_dataset \
    --dataset DAVIS2017val \
    --mode multi-object \
    --input-dir <result-dir>
    
```

Set `--dataset` and `--mode` accordingly. Again, adding `--eval-all` enables multi-run evaluation. The result table can
also be sorted by one of the metrics by adding `--sort-key <metric>`, e.g. `--sort-key J_and_F`

## Open source libraries

This code makes use of many open-source tools. The license files are placed in `./share/licenses`.

| Name              | License         | Repository                                          |
|-------------------|-----------------|-----------------------------------------------------|
| UDT               | MIT             | <https://github.com/594422814/UDT_pytorch>          |
| KCF,DSST          | MIT             | <https://github.com/fengyang95/pyCFTrackers>        |
| STC               | MIT             | <https://github.com/ajabri/videowalk>               |
| SWIN              | MIT             | <https://github.com/microsoft/Swin-Transformer>     |
| fhog + ColorTable | BSD             | <https://github.com/pdollar/toolbox>                |
| DAVIS2016 eval    | BSD             | <https://github.com/fperazzi/davis>                 |
| DINO              | Apache 2.0      | <https://github.com/facebookresearch/dino>          |
| TIMM              | Apache 2.0      | <https://github.com/rwightman/pytorch-image-models> |
| TimeCycle         | None            | <https://github.com/xiaolonw/TimeCycle>             |
| UVC               | None            | <https://github.com/Liusifei/UVC>                   |

## Cite us

    @inproceedings{waldmann2022improving,
      title={Improving Unsupervised Label Propagation for Pose Tracking and Video Object Segmentation},
      author={Waldmann, Urs and Bamberger, Jannik and Johannsen, Ole and Deussen, Oliver and Goldl\"{u}cke, Bastian},
      booktitle={DAGM German Conference on Pattern Recognition},
      year={2022},
      pages={230--245}
      }
