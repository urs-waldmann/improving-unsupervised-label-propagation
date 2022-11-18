import os
import sys

from os.path import join, dirname, abspath

# Storage_prefix is the base directory where all results and datasets are saved.
try:
    storage_prefix = os.environ['WORKSPACE_PREFIX']
except KeyError:
    # Instructions:
    # Simply set this path to the base directory you use. I needed this distinction, because it made it easier to
    # distinguish between my local machine and containers running remotely without setting an environment variable for
    # each of the ephemeral containers.
    storage_prefix = '../workspace/' if sys.platform != 'win32' else 'K:\\'

# Directory where all datasets are located
dataset_root = join(storage_prefix, 'datasets')
# Directory where checkpoints are saved
checkpoint_root = join(storage_prefix, 'checkpoints')
# Directory where experiment results are saved
results_root = join(storage_prefix, 'results')

# Find the source root directory relative to this file.
project_root = abspath(dirname(dirname(__file__)))

src_root = join(project_root, 'src')
share_root = join(project_root, 'share')
config_root = join(share_root, 'config')
labelprop_config_root = join(config_root, 'label_propagation')
config_schema_root = join(config_root, 'schema')

labelprop_schema_file = join(config_schema_root, 'config-schema.json')

ARG_PATH_MAPPINGS = {
    '$config': config_root,
    '$share': share_root,
}


def resolve_args_path(path):
    if path is None:
        return path

    for k, v in ARG_PATH_MAPPINGS.items():
        if path.startswith(k):
            return path.replace(k, v)

    return path


########################################################################################################################
# Dataset path definitions
########################################################################################################################

dataset_davis_root = join(dataset_root, 'DAVIS')
# Root of the DAVIS2016 dataset
dataset_davis2016_root = join(dataset_root, 'DAVIS', 'DAVIS2016')
dataset_davis2017_root = join(dataset_root, 'DAVIS', 'DAVIS2017')

# Root of the BSD500 dataset
dataset_bsds500_path = join(dataset_root, 'BSD', 'BSDS500')

# Path of the raw unprocessed ILSVRC2015_VID dataset
dataset_imagenet_path = join(dataset_root, 'ILSVRC2015')
# ImageNet preprocessed as in UDT
dataset_udt_imagenet_path = join(dataset_root, 'ILSVRC2015_VID_UDT')
# ImageNet preprocessed as in LUDT
dataset_ludt_imagenet_path = join(dataset_root, 'ILSVRC2015_VID_LUDT')

# Paths to the OTB100 dataset and manifest files for OTB50 and OTB100
dataset_otb_path = join(dataset_root, 'OTB')
dataset_otb2015_manifest_path = join(dataset_otb_path, f'OTB2015.json')
dataset_otb2013_manifest_path = join(dataset_otb_path, f'OTB2013.json')

# Root of the Pascal VOC2012 dataset
dataset_voc2012_path = join(dataset_root, 'VOC2012')

# Root of the JHMDB dataset
dataset_jhmdb_path = join(dataset_root, 'JHMDB')

# Root for the BADJA animal dataset
dataset_badja_path = join(dataset_root, 'BADJA')

# Root for the SegTrackV2 dataset
dataset_segtrackv2_path = join(dataset_root, 'SegTrackV2')
