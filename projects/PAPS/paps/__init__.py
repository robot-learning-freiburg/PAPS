# Copyright (c) Facebook, Inc. and its affiliates.
from .config import add_paps_config
from .dataset_mapper import PAPSAmodalDatasetMapper, PAPSDatasetMapper
from .panoptic_seg import (
    PAPS,
    INS_EMBED_BRANCHES_REGISTRY,
    build_ins_embed_branch,
    PAPSSemSegHead,
    PAPSInsEmbedHead,
)
