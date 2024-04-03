from copy import deepcopy
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import DeepLabV3PlusHead
from detectron2.projects.deeplab.loss import DeepLabCE
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.utils.registry import Registry

from .post_processing import get_panoptic_segmentation

__all__ = ["PAPS", "INS_EMBED_BRANCHES_REGISTRY", "build_ins_embed_branch"]


INS_EMBED_BRANCHES_REGISTRY = Registry("INS_EMBED_BRANCHES")
INS_EMBED_BRANCHES_REGISTRY.__doc__ = """
Registry for instance embedding branches, which make instance embedding
predictions from feature maps.
"""


@META_ARCH_REGISTRY.register()
class PAPS(nn.Module):
    """
    Main class for amodal/modal panoptic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.ins_embed_head = build_ins_embed_branch(cfg, self.backbone.output_shape())
        self.register_buffer(
            "pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False
        )
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.mode = cfg.MODEL.PAPS.MODE
        self.stuff_area = cfg.MODEL.PAPS.STUFF_AREA
        self.threshold = cfg.MODEL.PAPS.CENTER_THRESHOLD
        self.nms_kernel = cfg.MODEL.PAPS.NMS_KERNEL
        self.top_k = cfg.MODEL.PAPS.TOP_K_INSTANCE
        self.predict_instances = cfg.MODEL.PAPS.PREDICT_INSTANCES
        self.use_depthwise_separable_conv = cfg.MODEL.PAPS.USE_DEPTHWISE_SEPARABLE_CONV
        assert (
            cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
            == cfg.MODEL.PAPS.USE_DEPTHWISE_SEPARABLE_CONV
        )
        self.size_divisibility = cfg.MODEL.PAPS.SIZE_DIVISIBILITY
        self.benchmark_network_speed = cfg.MODEL.PAPS.BENCHMARK_NETWORK_SPEED

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * "center": center points heatmap ground truth
                   * "offset": pixel offsets to center points ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "panoptic_seg", "sem_seg": see documentation
                    :doc:`/tutorials/models` for the standard output format
                * "instances": available if ``predict_instances is True``. see documentation
                    :doc:`/tutorials/models` for the standard output format
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # To avoid error in ASPP layer when input has different size.
        size_divisibility = (
            self.size_divisibility
            if self.size_divisibility > 0
            else self.backbone.size_divisibility
        )
        images = ImageList.from_tensors(images, size_divisibility)

        features = self.backbone(images.tensor)

        if self.mode == "modal":
            return self.modal_forward(
                batched_inputs, features, images, size_divisibility
            )
        elif self.mode == "amodal":
            return self.amodal_forward(
                batched_inputs, features, images, size_divisibility
            )

    def modal_forward(self, batched_inputs, features, images, size_divisibility):
        losses = {}
        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
            if "sem_seg_weights" in batched_inputs[0]:
                # The default D2 DatasetMapper may not contain "sem_seg_weights"
                # Avoid error in testing when default DatasetMapper is used.
                weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                weights = ImageList.from_tensors(weights, size_divisibility).tensor
            else:
                weights = None
        else:
            targets = None
            weights = None

        sem_feats = self.sem_seg_head.layers_feat(features)
        sem_seg_results, sem_seg_losses = self.sem_seg_head(sem_feats, targets, weights)
        losses.update(sem_seg_losses)

        if "center" in batched_inputs[0] and "offset" in batched_inputs[0]:
            center_targets = [x["center"].to(self.device) for x in batched_inputs]
            center_targets = ImageList.from_tensors(
                center_targets, size_divisibility
            ).tensor.unsqueeze(1)
            center_weights = [
                x["center_weights"].to(self.device) for x in batched_inputs
            ]
            center_weights = ImageList.from_tensors(
                center_weights, size_divisibility
            ).tensor

            offset_targets = [x["offset"].to(self.device) for x in batched_inputs]
            offset_targets = ImageList.from_tensors(
                offset_targets, size_divisibility
            ).tensor
            offset_weights = [
                x["offset_weights"].to(self.device) for x in batched_inputs
            ]
            offset_weights = ImageList.from_tensors(
                offset_weights, size_divisibility
            ).tensor
        else:
            center_targets = None
            center_weights = None

            offset_targets = None
            offset_weights = None

        ins_feats = self.ins_embed_head.layers_feat(features)
        center_results, offset_results, center_losses, offset_losses = (
            self.ins_embed_head(
                ins_feats,
                center_targets,
                center_weights,
                offset_targets,
                offset_weights,
            )
        )
        losses.update(center_losses)
        losses.update(offset_losses)

        if self.training:
            return losses

        if self.benchmark_network_speed:
            return []

        processed_results = []
        for (
            sem_seg_result,
            center_result,
            offset_result,
            input_per_image,
            image_size,
        ) in zip(
            sem_seg_results,
            center_results,
            offset_results,
            batched_inputs,
            images.image_sizes,
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            c = sem_seg_postprocess(center_result, image_size, height, width)
            o = sem_seg_postprocess(offset_result, image_size, height, width)
            # Post-processing to get panoptic segmentation.
            panoptic_image, _ = get_panoptic_segmentation(
                r.argmax(dim=0, keepdim=True),
                c,
                o,
                thing_ids=self.meta.thing_dataset_id_to_contiguous_id.values(),
                label_divisor=self.meta.label_divisor,
                stuff_area=self.stuff_area,
                void_label=-1,
                threshold=self.threshold,
                nms_kernel=self.nms_kernel,
                top_k=self.top_k,
            )
            # For semantic segmentation evaluation.
            processed_results.append({"sem_seg": r})
            panoptic_image = panoptic_image.squeeze(0)
            semantic_prob = F.softmax(r, dim=0)
            # For panoptic segmentation evaluation.
            processed_results[-1]["panoptic_seg"] = (panoptic_image, None)
            # For instance segmentation evaluation.
            if self.predict_instances:
                instances = []
                panoptic_image_cpu = panoptic_image.cpu().numpy()
                for panoptic_label in np.unique(panoptic_image_cpu):
                    if panoptic_label == -1:
                        continue
                    pred_class = panoptic_label // self.meta.label_divisor
                    isthing = pred_class in list(
                        self.meta.thing_dataset_id_to_contiguous_id.values()
                    )
                    # Get instance segmentation results.
                    if isthing:
                        instance = Instances((height, width))
                        # Evaluation code takes continuous id starting from 0
                        instance.pred_classes = torch.tensor(
                            [pred_class], device=panoptic_image.device
                        )
                        mask = panoptic_image == panoptic_label
                        instance.pred_masks = mask.unsqueeze(0)
                        # Average semantic probability
                        sem_scores = semantic_prob[pred_class, ...]
                        sem_scores = torch.mean(sem_scores[mask])
                        # Center point probability
                        mask_indices = torch.nonzero(mask).float()
                        center_y, center_x = (
                            torch.mean(mask_indices[:, 0]),
                            torch.mean(mask_indices[:, 1]),
                        )
                        center_scores = c[0, int(center_y.item()), int(center_x.item())]
                        # Confidence score is semantic prob * center prob.
                        instance.scores = torch.tensor(
                            [sem_scores * center_scores], device=panoptic_image.device
                        )
                        # Get bounding boxes
                        instance.pred_boxes = BitMasks(
                            instance.pred_masks
                        ).get_bounding_boxes()
                        instances.append(instance)
                if len(instances) > 0:
                    processed_results[-1]["instances"] = Instances.cat(instances)

        return processed_results

    def amodal_forward(self, batched_inputs, features, images, size_divisibility):
        losses = {}
        if "sem_seg" in batched_inputs[0]:
            targets = {}
            targets["sem_seg"] = self.to_tensor(
                "sem_seg", batched_inputs, size_divisibility
            )
            targets["occ"] = self.to_tensor("occ", batched_inputs, size_divisibility)
            targets["occ_level_segs"] = self.to_tensor(
                "occ_level_segs", batched_inputs, size_divisibility
            )
            if "sem_seg_weights" in batched_inputs[0]:
                # The default D2 DatasetMapper may not contain "sem_seg_weights"
                # Avoid error in testing when default DatasetMapper is used.
                weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                weights = ImageList.from_tensors(weights, size_divisibility).tensor
            else:
                weights = None

        else:
            targets = None
            weights = None

        sem_feats, sem_occ_feats = self.sem_seg_head.layers_feat(features)
        sem_seg_results, sem_seg_losses = self.sem_seg_head(
            sem_feats, sem_occ_feats, targets, weights
        )
        losses.update(sem_seg_losses)

        if "center" in batched_inputs[0] and "offset" in batched_inputs[0]:
            center_targets = self.to_tensor("center", batched_inputs, size_divisibility)
            center_weights = self.to_tensor(
                "center_weights", batched_inputs, size_divisibility
            )
            offset_targets = self.to_tensor("offset", batched_inputs, size_divisibility)
            offset_weights = self.to_tensor(
                "offset_weights", batched_inputs, size_divisibility
            )
            amodal_offset_weights = self.to_tensor(
                "amodal_offset_weights", batched_inputs, size_divisibility
            )
            targets = {}
            targets["thing_seg"] = self.to_tensor(
                "thing_seg", batched_inputs, size_divisibility
            )
            if weights is not None:
                thing_weights = weights
            else:
                thing_weights = None
            targets["amodal_offsets"] = self.to_tensor(
                "amodal_offsets", batched_inputs, size_divisibility
            )
            targets["amodal_center_offsets"] = self.to_tensor(
                "amodal_center_offsets", batched_inputs, size_divisibility
            )

        else:
            center_targets = None
            center_weights = None

            targets = None
            thing_weights = None

            offset_targets = None
            offset_weights = None
            amodal_offset_weights = None

        ins_feats = self.ins_embed_head.layers_feat(features)
        (
            center_results,
            offset_results,
            other_results,
            center_losses,
            offset_losses,
            other_losses,
        ) = self.ins_embed_head(
            ins_feats,
            sem_occ_feats,
            center_targets,
            center_weights,
            offset_targets,
            offset_weights,
            targets,
            thing_weights,
            amodal_offset_weights,
        )
        losses.update(center_losses)
        losses.update(offset_losses)
        losses.update(other_losses)

        if self.training:
            return losses

        if self.benchmark_network_speed:
            return []

        processed_results = []
        for (
            sem_seg_result,
            center_result,
            offset_result,
            input_per_image,
            image_size,
        ) in zip(
            sem_seg_results["sem_seg"],
            center_results,
            offset_results,
            batched_inputs,
            images.image_sizes,
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            c = sem_seg_postprocess(center_result, image_size, height, width)
            o = sem_seg_postprocess(offset_result, image_size, height, width)
            # Post-processing to get panoptic segmentation.
            panoptic_image, _ = get_panoptic_segmentation(
                r.argmax(dim=0, keepdim=True),
                c,
                o,
                thing_ids=self.meta.thing_dataset_id_to_contiguous_id.values(),
                label_divisor=self.meta.label_divisor,
                stuff_area=self.stuff_area,
                void_label=-1,
                threshold=self.threshold,
                nms_kernel=self.nms_kernel,
                top_k=self.top_k,
            )
            # For semantic segmentation evaluation.
            processed_results.append({"sem_seg": r})
            panoptic_image = panoptic_image.squeeze(0)
            semantic_prob = F.softmax(r, dim=0)
            # For panoptic segmentation evaluation.
            processed_results[-1]["panoptic_seg"] = (panoptic_image, None)
            # For instance segmentation evaluation.
            if self.predict_instances:
                instances = []
                panoptic_image_cpu = panoptic_image.cpu().numpy()
                for panoptic_label in np.unique(panoptic_image_cpu):
                    if panoptic_label == -1:
                        continue
                    pred_class = panoptic_label // self.meta.label_divisor
                    isthing = pred_class in list(
                        self.meta.thing_dataset_id_to_contiguous_id.values()
                    )
                    # Get instance segmentation results.
                    if isthing:
                        instance = Instances((height, width))
                        # Evaluation code takes continuous id starting from 0
                        instance.pred_classes = torch.tensor(
                            [pred_class], device=panoptic_image.device
                        )
                        mask = panoptic_image == panoptic_label
                        instance.pred_masks = mask.unsqueeze(0)
                        # Average semantic probability
                        sem_scores = semantic_prob[pred_class, ...]
                        sem_scores = torch.mean(sem_scores[mask])
                        # Center point probability
                        mask_indices = torch.nonzero(mask).float()
                        center_y, center_x = (
                            torch.mean(mask_indices[:, 0]),
                            torch.mean(mask_indices[:, 1]),
                        )
                        center_scores = c[0, int(center_y.item()), int(center_x.item())]
                        # Confidence score is semantic prob * center prob.
                        instance.scores = torch.tensor(
                            [sem_scores * center_scores], device=panoptic_image.device
                        )
                        # Get bounding boxes
                        instance.pred_boxes = BitMasks(
                            instance.pred_masks
                        ).get_bounding_boxes()
                        instances.append(instance)
                if len(instances) > 0:
                    processed_results[-1]["instances"] = Instances.cat(instances)

        return processed_results

    def to_tensor(self, key, batched_inputs, size_divisibility):
        if key == "amodal_offset_weights":
            targets = []
            for i in range(1, self.sem_seg_head.num_occ_levels):
                targets_i = []
                for x in batched_inputs:
                    y = x[key]
                    y = y.to(self.device)
                    targets_i.append(y)
                targets_i = ImageList.from_tensors(targets_i, size_divisibility).tensor
                targets_i = torch.cat([targets_i, targets_i], dim=1)
                targets.append(targets_i)

        elif key == "amodal_offsets" or key == "amodal_center_offsets":
            targets = []
            for i in range(1, self.sem_seg_head.num_occ_levels):
                targets_i = []
                for x in batched_inputs:
                    amodal_level_offsets_x = x[key]

                    if "occ_level_{}".format(i) in amodal_level_offsets_x:
                        y = amodal_level_offsets_x["occ_level_{}".format(i)].to(
                            self.device
                        )
                    else:
                        y = (
                            torch.zeros_like(batched_inputs[0]["offset"])
                            + self.ins_embed_head.ignore_value
                        )
                        y = y.to(self.device)

                    targets_i.append(y)
                targets_i = ImageList.from_tensors(targets_i, size_divisibility).tensor
                targets.append(targets_i)

        elif "center" in key or "offset" in key:
            targets = [x[key].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(targets, size_divisibility).tensor
            if key == "center":
                targets = targets.unsqueeze(1)

        elif key == "occ_level_segs":
            targets = []
            for i in range(self.sem_seg_head.num_occ_levels):
                targets_i = []
                for x in batched_inputs:
                    occ_level_segs_x = x[key]

                    if "occ_level_{}".format(i) in occ_level_segs_x:
                        y = occ_level_segs_x["occ_level_{}".format(i)].to(self.device)
                    else:
                        y = (
                            torch.zeros_like(batched_inputs[0]["sem_seg"])
                            + self.sem_seg_head.ignore_value
                        )
                        y = y.to(self.device)

                    targets_i.append(y)
                targets_i = ImageList.from_tensors(
                    targets_i, size_divisibility, self.sem_seg_head.ignore_value
                ).tensor
                targets.append(targets_i)
        else:
            targets = [x[key].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        return targets


def build_ins_embed_branch(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.INS_EMBED_HEAD.NAME
    return INS_EMBED_BRANCHES_REGISTRY.get(name)(cfg, input_shape)


@SEM_SEG_HEADS_REGISTRY.register()
class PAPSSemDecoder(nn.Module):
    """
    A semantic segmentation head described in :paper:`Perceiving the Invisible: Proposal-Free Amodal Panoptic Segmentation`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        project_channels: List[int],
        dpc_dilations: List[int],
        decoder_channels: List[int],
        common_stride: int,
        norm: Union[str, Callable],
        loss_weight: float = 1.0,
        loss_type: str = "cross_entropy",
        ignore_value: int = -1,
        num_classes: Optional[int] = None,
        block_a_convs: int = 2,
        block_b_convs: int = 2,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shape of the input features. They will be ordered by stride
                and the last one (with largest stride) is used as the input to the
                decoder (i.e.  the ASPP module); the rest are low-level feature for
                the intermediate levels of decoder.
            project_channels (list[int]): a list of low-level feature channels.
                The length should be len(in_features) - 1.
            aspp_dilations (list(int)): a list of 3 dilations in ASPP.
            aspp_dropout (float): apply dropout on the output of ASPP.
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "in_features"
                (each element in "in_features" corresponds to one decoder stage).
            common_stride (int): output stride of decoder.
            norm (str or callable): normalization for all conv layers.
            train_size (tuple): (height, width) of training images.
            loss_weight (float): loss weight.
            loss_type (str): type of loss function, 2 opptions:
                (1) "cross_entropy" is the standard cross entropy loss.
                (2) "hard_pixel_mining" is the loss in DeepLab that samples
                    top k% hardest pixels.
            ignore_value (int): category to be ignored during training.
            num_classes (int): number of classes, if set to None, the decoder
                will not construct a predictor.
            use_depthwise_separable_conv (bool): use DepthwiseSeparableConv2d
                in ASPP and decoder.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)

        # fmt: off
        self.in_features      = [k for k, v in input_shape]  # starting from "res2" to "res5"
        in_channels           = [x[1].channels for x in input_shape]
        dpc_channels         = decoder_channels[-1] 
        self.ignore_value     = ignore_value
        self.common_stride    = common_stride  # output stride
        self.loss_weight      = loss_weight
        self.loss_type        = loss_type
        self.decoder_only     = num_classes is None
        self.block_a_convs = block_a_convs
        self.block_b_convs = block_b_convs
        # fmt: on

        assert (
            len(project_channels) == len(self.in_features) - 1
        ), "Expected {} project_channels, got {}".format(
            len(self.in_features) - 1, len(project_channels)
        )
        assert len(decoder_channels) == len(
            self.in_features
        ), "Expected {} decoder_channels, got {}".format(
            len(self.in_features), len(decoder_channels)
        )
        assert self.decoder_only, "PAPSSemDecoder is expected to be only decoder"

        self.decoder = nn.ModuleDict()

        self.project_conv = DPC(
            in_channels[-1] + in_channels[-2],
            dpc_channels,
            dpc_dilations,
            norm=norm,
            activation=F.relu,
        )

        self.convs = nn.ModuleList()
        for i in range(self.block_a_convs + self.block_b_convs):
            num_channels = (
                sum(in_channels) + dpc_channels
                if i == self.block_a_convs
                else dpc_channels
            )
            self.convs.append(
                DepthwiseSeparableConv2d(
                    num_channels,
                    dpc_channels,
                    kernel_size=3,
                    padding=1,
                    norm1=norm,
                    activation1=F.relu,
                    norm2=norm,
                    activation2=F.relu,
                )
            )

        if not self.decoder_only:
            self.predictor = Conv2d(
                decoder_channels[0], num_classes, kernel_size=1, stride=1, padding=0
            )
            nn.init.normal_(self.predictor.weight, 0, 0.001)
            nn.init.constant_(self.predictor.bias, 0)

            if self.loss_type == "cross_entropy":
                self.loss = nn.CrossEntropyLoss(
                    reduction="mean", ignore_index=self.ignore_value
                )
            elif self.loss_type == "hard_pixel_mining":
                self.loss = DeepLabCE(
                    ignore_label=self.ignore_value, top_k_percent_pixels=0.2
                )
            else:
                raise ValueError("Unexpected loss type: %s" % self.loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        decoder_channels = [cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.SEM_SEG_HEAD.DPC_CHANNELS]
        ret = dict(
            input_shape={
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS,
            dpc_dilations=cfg.MODEL.SEM_SEG_HEAD.DPC_DILATIONS,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.SEM_SEG_HEAD.NORM,
            loss_weight=cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            loss_type=cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE,
            ignore_value=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            block_a_convs=cfg.MODEL.SEM_SEG_HEAD.PROCESSING_BLOCK_A_CONVS,
            block_b_convs=cfg.MODEL.SEM_SEG_HEAD.PROCESSING_BLOCK_B_CONVS,
        )
        return ret

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(features)
        if self.decoder_only:
            # Output from self.layers() only contains decoder feature.
            return y
        if self.training:
            return None, self.losses(y, targets)
        else:
            y = F.interpolate(
                y, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return y, {}

    def layers(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        features = list(features.values())
        y = F.interpolate(
            features[-1],
            size=features[-2].size()[2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.project_conv(torch.cat([features[-2], y], dim=1))
        x = F.interpolate(
            x, size=features[-3].size()[2:], mode="bilinear", align_corners=False
        )
        for i in range(self.block_a_convs):
            x = self.convs[i](x)
        x = F.interpolate(
            x, size=features[-4].size()[2:], mode="bilinear", align_corners=False
        )

        for i in range(1, len(features)):
            features[i] = F.interpolate(
                features[i],
                size=features[-4].size()[2:],
                mode="bilinear",
                align_corners=False,
            )

        features.append(x)
        x = torch.cat(features, dim=1)
        for i in range(self.block_a_convs, self.block_a_convs + self.block_b_convs):
            x = self.convs[i](x)
        return x

    def losses(self, predictions, targets):
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        loss = self.loss(predictions, targets)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


@SEM_SEG_HEADS_REGISTRY.register()
class PAPSSemSegHead(PAPSSemDecoder):
    """
    A semantic segmentation head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        loss_weight: float,
        loss_type: str,
        loss_top_k: float,
        ignore_value: int,
        num_classes: int,
        cross_channels: int,
        use_depthwise_separable_conv: bool,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            ignore_value=ignore_value,
            **kwargs,
        )
        assert self.decoder_only

        self.loss_weight = loss_weight
        self.cross_channels = cross_channels
        self.use_depthwise_separable_conv = use_depthwise_separable_conv
        use_bias = norm == ""
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.head = DepthwiseSeparableConv2d(
                decoder_channels[0] + self.cross_channels,
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.head = nn.Sequential(
                Conv2d(
                    decoder_channels[0] + self.cross_channels,
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.head[0])
            weight_init.c2_xavier_fill(self.head[1])

        self.predictor = Conv2d(head_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        if loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_value)
        elif loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(
                ignore_label=ignore_value, top_k_percent_pixels=loss_top_k
            )
        else:
            raise ValueError("Unexpected loss type: %s" % loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS
        ret["loss_top_k"] = cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K
        ret["cross_channels"] = cfg.MODEL.SEM_SEG_HEAD.CROSS_CHANNELS
        ret["use_depthwise_separable_conv"] = (
            cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
        )
        return ret

    def forward(self, features, targets=None, weights=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers_out(features)
        if self.training:
            return None, self.losses(y, targets, weights)
        else:
            y = F.interpolate(
                y, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return y, {}

    def layers_feat(self, features):
        assert self.decoder_only
        y = super().layers(features)
        return y

    def layers_out(self, features):
        assert self.decoder_only
        y = features
        y = self.head(y)
        y = self.predictor(y)
        return y

    def losses(self, predictions, targets, weights=None):
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        loss = self.loss(predictions, targets, weights)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


@SEM_SEG_HEADS_REGISTRY.register()
class PAPSAmSemSegHead(PAPSSemDecoder):
    """
    A semantic segmentation head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        loss_weight: float,
        loss_type: str,
        loss_top_k: float,
        ignore_value: int,
        num_classes: int,
        cross_channels: int,
        use_depthwise_separable_conv: bool,
        num_occ_levels: int,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            ignore_value=ignore_value,
            **kwargs,
        )
        assert self.decoder_only

        self.loss_weight = loss_weight
        self.cross_channels = cross_channels
        self.use_depthwise_separable_conv = use_depthwise_separable_conv
        self.num_occ_levels = num_occ_levels
        # `head` is additional transform before predictor
        self.sem_seg_head = PAPSHead(
            decoder_channels[0],
            decoder_channels[0],
            head_channels,
            norm,
            activation=F.relu,
        )
        self.sem_seg_predictor = Conv2d(head_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.sem_seg_predictor.weight, 0, 0.001)
        nn.init.constant_(self.sem_seg_predictor.bias, 0)

        self.rel_occ_ord_seg_head = nn.ModuleList()
        self.rel_occ_ord_seg_predictor = nn.ModuleList()
        for i in range(self.num_occ_levels):
            self.rel_occ_ord_seg_head.append(
                PAPSHead(
                    decoder_channels[0] * 2,
                    decoder_channels[0] // 2,
                    head_channels // 2,
                    norm,
                    activation=F.relu,
                )
            )
            self.rel_occ_ord_seg_predictor.append(
                Conv2d(head_channels // 2, 1, kernel_size=1)
            )
            nn.init.normal_(self.rel_occ_ord_seg_predictor[i].weight, 0, 0.001)
            nn.init.constant_(self.rel_occ_ord_seg_predictor[i].bias, 0)

        self.occ_head = PAPSHead(
            decoder_channels[0],
            decoder_channels[0] // 2,
            head_channels // 2,
            norm,
            activation=F.relu,
        )
        self.occ_predictor = Conv2d(head_channels // 2, 1, kernel_size=1)
        nn.init.normal_(self.occ_predictor.weight, 0, 0.001)
        nn.init.constant_(self.occ_predictor.bias, 0)

        if loss_type == "cross_entropy":
            self.seg_loss = nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_value
            )
        elif loss_type == "hard_pixel_mining":
            self.seg_loss = DeepLabCE(
                ignore_label=ignore_value, top_k_percent_pixels=loss_top_k
            )
        else:
            raise ValueError("Unexpected loss type: %s" % loss_type)
        self.occ_loss = nn.BCEWithLogitsLoss(reduction="mean")

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS
        ret["loss_top_k"] = cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K
        ret["cross_channels"] = cfg.MODEL.SEM_SEG_HEAD.CROSS_CHANNELS
        ret["use_depthwise_separable_conv"] = (
            cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
        )
        ret["num_occ_levels"] = cfg.MODEL.SEM_SEG_HEAD.NUM_OCC_LEVELS
        return ret

    def forward(self, features, occ_features, targets=None, weights=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y_seg, y_occ, y_rel_occ_ord = self.layers_out(features, occ_features)
        if self.training:
            return None, self.losses(y_seg, y_occ, y_rel_occ_ord, targets, weights)
        else:
            return self.inference(y_seg, y_occ, y_rel_occ_ord), {}

    def layers_feat(self, features):
        assert self.decoder_only
        y = super().layers(features)
        y_occ = self.occ_head(y)
        return y, y_occ

    def layers_out(self, features, y_occ):
        assert self.decoder_only
        y = features
        y_seg = self.sem_seg_head(y)
        y_seg_pred = self.sem_seg_predictor(y_seg)

        y_occ_pred = self.occ_predictor(y_occ)

        y_rel_occ_ord_pred = []
        for i in range(len(self.rel_occ_ord_seg_head)):
            if i == 0:
                y_rel_occ_ord = torch.cat([y, y_seg], dim=1)
            else:
                y_rel_occ_ord = torch.cat([y_rel_occ_ord, y_seg, y_occ], dim=1)
            y_rel_occ_ord = self.rel_occ_ord_seg_head[i](y_rel_occ_ord)
            y_rel_occ_ord_pred.append(self.rel_occ_ord_seg_predictor[i](y_rel_occ_ord))

        return y_seg_pred, y_occ_pred, y_rel_occ_ord_pred

    def losses(
        self,
        seg_predictions,
        occ_predictions,
        rel_occ_ord_predictions,
        targets,
        weights=None,
    ):
        seg_predictions = self.upscale(seg_predictions)
        occ_predictions = self.upscale(occ_predictions)
        for i in range(len(rel_occ_ord_predictions)):
            rel_occ_ord_predictions[i] = self.upscale(rel_occ_ord_predictions[i])

        seg_loss = self.seg_loss(seg_predictions, targets["sem_seg"], weights)
        occ_loss = self.occ_loss(occ_predictions.squeeze(1), targets["occ"].float())

        rel_occ_ord_batch = torch.cat(rel_occ_ord_predictions, dim=0).squeeze(1)
        target_rel_occ_ord_batch = torch.cat(targets["occ_level_segs"], dim=0)
        valid_mask = target_rel_occ_ord_batch != self.ignore_value
        masked_rel_occ_ord_batch = rel_occ_ord_batch[valid_mask]
        masked_target_rel_occ_ord_batch = target_rel_occ_ord_batch[valid_mask]

        if masked_target_rel_occ_ord_batch.numel() == 0:
            rel_occ_loss = torch.tensor(1e-4, requires_grad=True)
        else:
            rel_occ_loss = self.occ_loss(
                masked_rel_occ_ord_batch, masked_target_rel_occ_ord_batch.float()
            )

        losses = {
            "loss_sem_seg": seg_loss * self.loss_weight,
            "loss_occ": occ_loss * self.loss_weight,
            "loss_rel_occ_ord": rel_occ_loss * self.loss_weight,
        }
        return losses

    def upscale(self, x):
        return F.interpolate(
            x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )

    def inference(self, seg_predictions, occ_predictions, rel_occ_ord_predictions):
        seg_predictions = self.upscale(seg_predictions)
        occ_predictions = self.upscale(occ_predictions)
        for i in range(len(rel_occ_ord_predictions)):
            rel_occ_ord_predictions[i] = self.upscale(rel_occ_ord_predictions[i])
        preds = {
            "sem_seg": seg_predictions,
            "occ": occ_predictions,
            "rel_occ_ord": rel_occ_ord_predictions,
        }
        return preds


@SEM_SEG_HEADS_REGISTRY.register()
class PAPSInsDecoder(nn.Module):
    """
    A semantic segmentation head described in :paper:`DeepLabV3+`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        project_channels: List[int],
        ce_dilations: List[int],
        decoder_channels: List[int],
        common_stride: int,
        norm: Union[str, Callable],
        loss_weight: float = 1.0,
        loss_type: str = "cross_entropy",
        ignore_value: int = -1,
        num_classes: Optional[int] = None,
        block_a_convs: int = 2,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shape of the input features. They will be ordered by stride
                and the last one (with largest stride) is used as the input to the
                decoder (i.e.  the ASPP module); the rest are low-level feature for
                the intermediate levels of decoder.
            project_channels (list[int]): a list of low-level feature channels.
                The length should be len(in_features) - 1.
            aspp_dilations (list(int)): a list of 3 dilations in ASPP.
            aspp_dropout (float): apply dropout on the output of ASPP.
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "in_features"
                (each element in "in_features" corresponds to one decoder stage).
            common_stride (int): output stride of decoder.
            norm (str or callable): normalization for all conv layers.
            train_size (tuple): (height, width) of training images.
            loss_weight (float): loss weight.
            loss_type (str): type of loss function, 2 opptions:
                (1) "cross_entropy" is the standard cross entropy loss.
                (2) "hard_pixel_mining" is the loss in DeepLab that samples
                    top k% hardest pixels.
            ignore_value (int): category to be ignored during training.
            num_classes (int): number of classes, if set to None, the decoder
                will not construct a predictor.
            use_depthwise_separable_conv (bool): use DepthwiseSeparableConv2d
                in ASPP and decoder.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)

        # fmt: off
        self.in_features      = [k for k, v in input_shape]  # starting from "res2" to "res5"
        in_channels           = [x[1].channels for x in input_shape]
        ce_channels           = decoder_channels[-1] 
        self.num_scales       = len(in_channels)
        self.ignore_value     = ignore_value
        self.common_stride    = common_stride  # output stride
        self.loss_weight      = loss_weight
        self.loss_type        = loss_type
        self.block_a_convs    = block_a_convs
        self.decoder_only     = num_classes is None
        # fmt: on

        assert (
            len(project_channels) == len(self.in_features) - 1
        ), "Expected {} project_channels, got {}".format(
            len(self.in_features) - 1, len(project_channels)
        )
        assert len(decoder_channels) == len(
            self.in_features
        ), "Expected {} decoder_channels, got {}".format(
            len(self.in_features), len(decoder_channels)
        )
        assert self.decoder_only, "PAPSInsDecoder is expected to be only decoder"

        self.decoder = nn.ModuleDict()

        self.convs = nn.ModuleList()
        self.proj_convs = nn.ModuleList()
        c_channels = [sum(in_channels), ce_channels, ce_channels, ce_channels]

        for i in range(self.num_scales):
            self.convs.append(
                CE(
                    in_channels[i],
                    ce_channels,
                    ce_dilations,
                    norm=norm,
                    activation=F.relu,
                )
            )

            self.proj_convs.append(
                DepthwiseSeparableConv2d(
                    c_channels[i],
                    ce_channels,
                    kernel_size=3,
                    padding=1,
                    norm1=norm,
                    activation1=F.relu,
                    norm2=norm,
                    activation2=F.relu,
                )
            )

        self.pre_fuse_convs = nn.ModuleList()
        for i in range(self.num_scales):
            block_convs = nn.ModuleList()
            for j in range(self.block_a_convs):
                num_in_channels = ce_channels if i == 0 or j != 0 else ce_channels * 2
                num_out_channels = (
                    128
                    if i == self.num_scales - 1 and j == self.block_a_convs - 1
                    else ce_channels
                )
                block_convs.append(
                    DepthwiseSeparableConv2d(
                        num_in_channels,
                        num_out_channels,
                        kernel_size=3,
                        padding=1,
                        norm1=norm,
                        activation1=F.relu,
                        norm2=norm,
                        activation2=F.relu,
                    )
                )
            self.pre_fuse_convs.append(block_convs)

        if not self.decoder_only:
            self.predictor = Conv2d(
                decoder_channels[0], num_classes, kernel_size=1, stride=1, padding=0
            )
            nn.init.normal_(self.predictor.weight, 0, 0.001)
            nn.init.constant_(self.predictor.bias, 0)

            if self.loss_type == "cross_entropy":
                self.loss = nn.CrossEntropyLoss(
                    reduction="mean", ignore_index=self.ignore_value
                )
            elif self.loss_type == "hard_pixel_mining":
                self.loss = DeepLabCE(
                    ignore_label=self.ignore_value, top_k_percent_pixels=0.2
                )
            else:
                raise ValueError("Unexpected loss type: %s" % self.loss_type)

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(features)
        if self.decoder_only:
            # Output from self.layers() only contains decoder feature.
            return y
        if self.training:
            return None, self.losses(y, targets)
        else:
            y = F.interpolate(
                y, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return y, {}

    def layers(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        features = list(features.values())
        features_size = []
        for i in range(self.num_scales):
            features_size.append(features[i].size()[2:])

        x_s = []
        for i in range(self.num_scales):
            x_s.append(self.convs[i](features[i]))

        for i in range(1, self.num_scales):
            features[i] = F.interpolate(
                features[i],
                size=features[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat(features, dim=1)
        x = self.proj_convs[0](x)

        x_scales = [
            x,
        ]
        for i in range(1, self.num_scales):
            x_scales.append(
                F.interpolate(
                    x_scales[-1],
                    size=features_size[i],
                    mode="bilinear",
                    align_corners=False,
                )
            )
        x_scales[0] = x_scales[0] + x_s[0]
        for i in range(1, self.num_scales):
            x_scales[i] = self.proj_convs[i](x_scales[i]) + x_s[i]

        for i in range(self.num_scales):
            if i < self.num_scales - 1:
                x3 = self.pre_fuse_convs[i][0](x_scales[self.num_scales - 1 - i])
                for conv in self.pre_fuse_convs[i][1:]:
                    x3 = conv(x3)
                x3 = F.interpolate(
                    x3,
                    size=features_size[self.num_scales - 2 - i],
                    mode="bilinear",
                    align_corners=False,
                )
                x_scales[self.num_scales - 2 - i] = torch.cat(
                    [x_scales[self.num_scales - 2 - i], x3], dim=1
                )
            else:
                x = self.pre_fuse_convs[i][0](x_scales[self.num_scales - 1 - i])
                for conv in self.pre_fuse_convs[i][1:]:
                    x = conv(x)
        return x

    def losses(self, predictions, targets):
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        loss = self.loss(predictions, targets)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


@INS_EMBED_BRANCHES_REGISTRY.register()
class PAPSInsEmbedHead(PAPSInsDecoder):
    """
    A instance embedding head described in :paper:`PAPS`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        center_loss_weight: float,
        offset_loss_weight: float,
        use_depthwise_separable_conv: bool,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            center_loss_weight (float): loss weight for center point prediction.
            offset_loss_weight (float): loss weight for center offset prediction.
        """
        super().__init__(
            input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs
        )
        assert self.decoder_only

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        self.use_depthwise_separable_conv = use_depthwise_separable_conv
        use_bias = norm == ""
        # center prediction
        # `head` is additional transform before predictor
        self.center_head = nn.Sequential(
            Conv2d(
                decoder_channels[0] + kwargs["cross_channels"],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )
        weight_init.c2_xavier_fill(self.center_head[0])
        weight_init.c2_xavier_fill(self.center_head[1])
        self.center_predictor = Conv2d(head_channels, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        # offset prediction
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.offset_head = DepthwiseSeparableConv2d(
                decoder_channels[0] + kwargs["cross_channels"],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.offset_head = nn.Sequential(
                Conv2d(
                    decoder_channels[0] + kwargs["cross_channels"],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.offset_head[0])
            weight_init.c2_xavier_fill(self.offset_head[1])
        self.offset_predictor = Conv2d(head_channels, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)

        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")

    @classmethod
    def from_config(cls, cfg, input_shape):
        decoder_channels = [cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.INS_EMBED_HEAD.CE_CHANNELS]
        ret = dict(
            input_shape={
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS,
            ce_dilations=cfg.MODEL.INS_EMBED_HEAD.CE_DILATIONS,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.INS_EMBED_HEAD.NORM,
            head_channels=cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS,
            center_loss_weight=cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT,
            offset_loss_weight=cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
            block_a_convs=cfg.MODEL.INS_EMBED_HEAD.BLOCK_A_CONVS,
        )
        ret["cross_channels"] = cfg.MODEL.INS_EMBED_HEAD.CROSS_CHANNELS
        ret["use_depthwise_separable_conv"] = (
            cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
        )
        return ret

    def forward(
        self,
        features,
        center_targets=None,
        center_weights=None,
        offset_targets=None,
        offset_weights=None,
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        center, offset = self.layers_out(features)
        if self.training:
            return (
                None,
                None,
                self.center_losses(center, center_targets, center_weights),
                self.offset_losses(offset, offset_targets, offset_weights),
            )
        else:
            center = F.interpolate(
                center,
                scale_factor=self.common_stride,
                mode="bilinear",
                align_corners=False,
            )
            offset = (
                F.interpolate(
                    offset,
                    scale_factor=self.common_stride,
                    mode="bilinear",
                    align_corners=False,
                )
                * self.common_stride
            )
            return center, offset, {}, {}

    def layers_feat(self, features):
        assert self.decoder_only
        y = super().layers(features)
        return y

    def layers_out(self, features):
        assert self.decoder_only
        y = features
        # center
        center = self.center_head(y)
        center = self.center_predictor(center)
        # offset
        offset = self.offset_head(y)
        offset = self.offset_predictor(offset)
        return center, offset

    def center_losses(self, predictions, targets, weights):
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        loss = self.center_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses

    def offset_losses(self, predictions, targets, weights):
        predictions = (
            F.interpolate(
                predictions,
                scale_factor=self.common_stride,
                mode="bilinear",
                align_corners=False,
            )
            * self.common_stride
        )
        loss = self.offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_offset": loss * self.offset_loss_weight}
        return losses


@INS_EMBED_BRANCHES_REGISTRY.register()
class PAPSAmInsEmbedHead(PAPSInsDecoder):
    """
    A instance embedding head described in :paper:`PAPS`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        center_loss_weight: float,
        offset_loss_weight: float,
        use_depthwise_separable_conv: bool,
        num_occ_levels: int,
        num_thing_classes: int,
        ignore_value: int,
        ignore_offset_value: int,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            center_loss_weight (float): loss weight for center point prediction.
            offset_loss_weight (float): loss weight for center offset prediction.
        """
        super().__init__(
            input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs
        )
        assert self.decoder_only

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        self.use_depthwise_separable_conv = use_depthwise_separable_conv
        self.ignore_value = ignore_value
        self.ignore_offset_value = ignore_offset_value
        self.num_occ_levels = num_occ_levels
        use_bias = norm == ""
        # center prediction
        # `head` is additional transform before predictor
        self.thing_head = PAPSHead(
            decoder_channels[0] + decoder_channels[0] // 2,
            decoder_channels[0],
            head_channels,
            norm,
            activation=F.relu,
        )
        self.thing_predictor = Conv2d(head_channels, num_thing_classes, kernel_size=1)
        nn.init.normal_(self.thing_predictor.weight, 0, 0.001)
        nn.init.constant_(self.thing_predictor.bias, 0)

        self.center_head = PAPSHead(
            decoder_channels[0] + decoder_channels[0] // 2,
            decoder_channels[0],
            head_channels,
            norm,
            activation=F.relu,
        )
        self.center_predictor = Conv2d(head_channels, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        self.offset_head = PAPSHead(
            decoder_channels[0] + decoder_channels[0] // 2,
            decoder_channels[0],
            head_channels,
            norm,
            activation=F.relu,
        )
        self.offset_predictor = Conv2d(head_channels, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)

        self.amodal_center_offset_head = nn.ModuleList()
        self.amodal_center_offset_predictor = nn.ModuleList()
        self.amodal_offset_head = nn.ModuleList()
        self.amodal_offset_predictor = nn.ModuleList()
        for i in range(self.num_occ_levels - 1):
            self.amodal_offset_head.append(
                PAPSHead(
                    (decoder_channels[0] * 7) // 4,
                    decoder_channels[0],
                    head_channels,
                    norm,
                    activation=F.relu,
                )
            )
            self.amodal_offset_predictor.append(Conv2d(head_channels, 2, kernel_size=1))
            nn.init.normal_(self.amodal_offset_predictor[i].weight, 0, 0.001)
            nn.init.constant_(self.amodal_offset_predictor[i].bias, 0)
            self.amodal_center_offset_head.append(
                PAPSHead(
                    (decoder_channels[0] * 7) // 4,
                    decoder_channels[0],
                    head_channels,
                    norm,
                    activation=F.relu,
                )
            )
            self.amodal_center_offset_predictor.append(
                Conv2d(head_channels, 2, kernel_size=1)
            )
            nn.init.normal_(self.amodal_center_offset_predictor[i].weight, 0, 0.001)
            nn.init.constant_(self.amodal_center_offset_predictor[i].bias, 0)

        self.project_occ = Conv2d(
            decoder_channels[0],
            decoder_channels[0] // 2,
            kernel_size=1,
            norm=get_norm(norm, decoder_channels[0] // 2),
            activation=F.relu,
        )
        nn.init.normal_(self.project_occ.weight, 0, 0.001)
        nn.init.constant_(self.project_occ.bias, 0)

        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")
        self.thing_loss = DeepLabCE(ignore_label=255, top_k_percent_pixels=0.2)

    @classmethod
    def from_config(cls, cfg, input_shape):
        decoder_channels = [cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.INS_EMBED_HEAD.CE_CHANNELS]
        ret = dict(
            input_shape={
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS,
            ce_dilations=cfg.MODEL.INS_EMBED_HEAD.CE_DILATIONS,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.INS_EMBED_HEAD.NORM,
            head_channels=cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS,
            center_loss_weight=cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT,
            offset_loss_weight=cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
            block_a_convs=cfg.MODEL.INS_EMBED_HEAD.BLOCK_A_CONVS,
        )
        ret["cross_channels"] = cfg.MODEL.INS_EMBED_HEAD.CROSS_CHANNELS
        ret["use_depthwise_separable_conv"] = (
            cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
        )
        ret["num_thing_classes"] = cfg.MODEL.INS_EMBED_HEAD.NUM_THING_CLASSES
        ret["num_occ_levels"] = cfg.MODEL.SEM_SEG_HEAD.NUM_OCC_LEVELS
        ret["ignore_value"] = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        ret["ignore_offset_value"] = cfg.MODEL.INS_EMBED_HEAD.IGNORE_VALUE

        return ret

    def forward(
        self,
        features,
        occ_features,
        center_targets=None,
        center_weights=None,
        offset_targets=None,
        offset_weights=None,
        other_targets=None,
        thing_weights=None,
        amodal_offset_weights=None,
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        center, offset, thing, amodal_offsets, amodal_center_offsets = self.layers_out(
            features, occ_features
        )
        if self.training:
            return (
                None,
                None,
                None,
                self.center_losses(center, center_targets, center_weights),
                self.offset_losses(offset, offset_targets, offset_weights),
                self.other_losses(
                    thing,
                    amodal_offsets,
                    amodal_center_offsets,
                    other_targets,
                    thing_weights,
                    amodal_offset_weights,
                ),
            )
        else:
            center = F.interpolate(
                center,
                scale_factor=self.common_stride,
                mode="bilinear",
                align_corners=False,
            )
            offset = (
                F.interpolate(
                    offset,
                    scale_factor=self.common_stride,
                    mode="bilinear",
                    align_corners=False,
                )
                * self.common_stride
            )

            others = self.get_other_results(
                thing, amodal_offsets, amodal_center_offsets
            )

            return center, offset, others, {}, {}, {}

    def layers_feat(self, features):
        assert self.decoder_only
        y = super().layers(features)
        return y

    def layers_out(self, features, occ_features):
        assert self.decoder_only
        occ_features = self.project_occ(occ_features)
        y = torch.cat([features, occ_features], dim=1)
        # offset
        offset = self.offset_head(y)
        offset = self.offset_predictor(offset)
        # thing
        thing_feat = self.thing_head(y)
        thing = self.thing_predictor(thing_feat)
        # center
        center_feat = self.center_head(y)
        center = self.center_predictor(center_feat)

        y_amodal_offsets = []
        y_amodal_center_offsets = []
        for i in range(self.num_occ_levels - 1):
            if i == 0:
                y_amodal_offset = torch.cat([y, thing_feat], dim=1)
                y_amodal_center_offset = torch.cat([y, center_feat], dim=1)
            else:
                y_amodal_offset = torch.cat([y, y_amodal_offset], dim=1)
                y_amodal_center_offset = torch.cat([y, y_amodal_center_offset], dim=1)
            # amodal offset
            y_amodal_offset = self.amodal_offset_head[i](y_amodal_offset)
            y_amodal_offsets.append(self.amodal_offset_predictor[i](y_amodal_offset))
            # amodal center offset
            y_amodal_center_offset = self.amodal_center_offset_head[i](
                y_amodal_center_offset
            )
            y_amodal_center_offsets.append(
                self.amodal_center_offset_predictor[i](y_amodal_center_offset)
            )

        return center, offset, thing, y_amodal_offsets, y_amodal_center_offsets

    def center_losses(self, predictions, targets, weights):
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        loss = self.center_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses

    def offset_losses(self, predictions, targets, weights):
        predictions = (
            F.interpolate(
                predictions,
                scale_factor=self.common_stride,
                mode="bilinear",
                align_corners=False,
            )
            * self.common_stride
        )
        loss = self.offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_offset": loss * self.offset_loss_weight}
        return losses

    def other_losses(
        self,
        thing_predictions,
        amodal_offsets,
        amodal_center_offsets,
        targets,
        thing_weights,
        amodal_weights,
    ):
        thing_predictions = self.upscale(thing_predictions)
        thing_loss = self.thing_loss(
            thing_predictions, targets["thing_seg"], thing_weights
        )

        amodal_offsets = self.offset_upscale(torch.cat(amodal_offsets, dim=0))
        amodal_center_offsets = self.offset_upscale(
            torch.cat(amodal_center_offsets, dim=0)
        )

        target_amodal_offsets = torch.cat(targets["amodal_offsets"], dim=0)
        amodal_weights = torch.cat(amodal_weights, dim=0)
        valid_mask = target_amodal_offsets != self.ignore_offset_value
        masked_amodal_offsets = amodal_offsets[valid_mask]
        masked_target_amodal_offsets = target_amodal_offsets[valid_mask]
        if masked_target_amodal_offsets.numel() == 0 or amodal_weights.sum() == 0:
            amodal_offset_loss = torch.tensor(1e-4, requires_grad=True)
        else:
            amodal_offset_loss = (
                self.offset_loss(masked_amodal_offsets, masked_target_amodal_offsets)
                * amodal_weights[valid_mask]
            )
            amodal_offset_loss = (
                amodal_offset_loss.sum() / amodal_weights[valid_mask].sum()
            )

        target_amodal_center_offsets = torch.cat(
            targets["amodal_center_offsets"], dim=0
        )
        masked_amodal_center_offsets = amodal_center_offsets[valid_mask]
        masked_target_amodal_center_offsets = target_amodal_center_offsets[valid_mask]
        if (
            masked_target_amodal_center_offsets.numel() == 0
            or amodal_weights.sum() == 0
        ):
            amodal_center_offset_loss = torch.tensor(1e-4, requires_grad=True)
        else:
            amodal_center_offset_loss = (
                self.offset_loss(
                    masked_amodal_center_offsets, masked_target_amodal_center_offsets
                )
                * amodal_weights[valid_mask]
            )
            amodal_center_offset_loss = (
                amodal_center_offset_loss.sum() / amodal_weights[valid_mask].sum()
            )

        losses = {
            "loss_thing_seg": thing_loss,
            "loss_amodal_offset": amodal_offset_loss,
            "loss_amodal_center_offset": amodal_center_offset_loss,
        }

        return losses

    def upscale(self, x):
        return F.interpolate(
            x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )

    def offset_upscale(self, x):
        return (
            F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            * self.common_stride
        )

    def get_other_results(self, thing, amodal_offsets, amodal_center_offsets):
        thing = self.upscale(thing)
        amodal_offsets = self.offset_upscale(torch.cat(amodal_offsets, dim=0))
        amodal_offsets = torch.chunk(amodal_offsets, self.num_occ_levels - 1, dim=0)
        amodal_center_offsets = self.offset_upscale(
            torch.cat(amodal_center_offsets, dim=0)
        )
        amodal_center_offsets = torch.chunk(
            amodal_center_offsets, self.num_occ_levels - 1, dim=0
        )
        other_results = {
            "thing_seg": thing,
            "amodal_offsets": amodal_offsets,
            "amodal_center_offsets": amodal_center_offsets,
        }
        return other_results


class PAPSHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        decoder_channels: int,
        head_channels: int,
        norm,
        activation,
    ):

        super(PAPSHead, self).__init__()
        self.paps_head = nn.Sequential(
            DepthwiseSeparableConv2d(
                in_channels,
                decoder_channels,
                norm1=norm,
                activation1=deepcopy(activation),
                norm2=norm,
                activation2=deepcopy(activation),
            ),
            DepthwiseSeparableConv2d(
                decoder_channels,
                head_channels,
                norm1=norm,
                activation1=deepcopy(activation),
                norm2=norm,
                activation2=deepcopy(activation),
            ),
        )

    def forward(self, x):
        return self.paps_head(x)


class DPC(nn.Module):
    """
    Dense Prediction Cell (DPC).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dilations,
        *,
        norm,
        activation,
    ):
        """
        Args:
            in_channels (int): number of input channels for ASPP.
            out_channels (int): number of output channels.
            dilations (list): a list of 3 dilations in ASPP.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format. norm is
                applied to all conv layers except the conv following
                global average pooling.
            activation (callable): activation function.
            pool_kernel_size (tuple, list): the average pooling size (kh, kw)
                for image pooling layer in ASPP. If set to None, it always
                performs global average pooling. If not None, it must be
                divisible by the shape of inputs in forward(). It is recommended
                to use a fixed input feature size in training, and set this
                option to match this size, so that it performs global average
                pooling in training, and the size of the pooling window stays
                consistent in inference.
            dropout (float): apply dropout on the output of ASPP. It is used in
                the official DeepLab implementation with a rate of 0.1:
                https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/model.py#L532  # noqa
            use_depthwise_separable_conv (bool): use DepthwiseSeparableConv2d
                for 3x3 convs in ASPP, proposed in :paper:`DeepLabV3+`.
        """
        super(DPC, self).__init__()
        assert len(dilations) == 5, "DPC expects 5 dilations, got {}".format(
            len(dilations)
        )
        # dilations = [(1,6), (1,1), (6,21), (18,15), (6,3)]
        use_bias = norm == ""
        self.convs = nn.ModuleList()

        self.convs.append(
            DepthwiseSeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                norm1=norm,
                activation1=deepcopy(activation),
                norm2=norm,
                activation2=deepcopy(activation),
            )
        )
        # atrous convs
        for dilation in dilations[1:]:
            self.convs.append(
                DepthwiseSeparableConv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    norm1=norm,
                    activation1=deepcopy(activation),
                    norm2=norm,
                    activation2=deepcopy(activation),
                )
            )

        self.project = Conv2d(
            len(dilations) * out_channels,
            out_channels,
            kernel_size=1,
            bias=use_bias,
            norm=get_norm(norm, out_channels),
            activation=deepcopy(activation),
        )
        weight_init.c2_xavier_fill(self.project)

    def forward(self, x):
        res = []
        x = self.convs[0](x)
        res.append(x)
        for conv in self.convs[1:-1]:
            res.append(conv(x))
        res.append(self.convs[-1](res[-1]))
        res = torch.cat(res, dim=1)
        res = self.project(res)
        return res


class CE(nn.Module):
    """
    Context Extractor Module (CE).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dilations,
        *,
        norm,
        activation,
    ):
        """
        Args:
            in_channels (int): number of input channels for ASPP.
            out_channels (int): number of output channels.
            dilations (list): a list of 3 dilations in ASPP.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format. norm is
                applied to all conv layers except the conv following
                global average pooling.
            activation (callable): activation function.
            pool_kernel_size (tuple, list): the average pooling size (kh, kw)
                for image pooling layer in ASPP. If set to None, it always
                performs global average pooling. If not None, it must be
                divisible by the shape of inputs in forward(). It is recommended
                to use a fixed input feature size in training, and set this
                option to match this size, so that it performs global average
                pooling in training, and the size of the pooling window stays
                consistent in inference.
            dropout (float): apply dropout on the output of ASPP. It is used in
                the official DeepLab implementation with a rate of 0.1:
                https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/model.py#L532  # noqa
            use_depthwise_separable_conv (bool): use DepthwiseSeparableConv2d
                for 3x3 convs in ASPP, proposed in :paper:`DeepLabV3+`.
        """
        super(CE, self).__init__()
        assert len(dilations) == 2, "CE expects 2 dilations, got {}".format(
            len(dilations)
        )
        # dilations = [(1,6), (3,1)]
        self.gap = nn.AdaptiveAvgPool2d(1)
        use_bias = norm == ""
        self.convs = nn.ModuleList()

        self.convs.append(
            DepthwiseSeparableConv2d(
                in_channels,
                out_channels // 2,
                kernel_size=3,
                padding=(1, 1),
                dilation=(1, 1),
                norm1=norm,
                activation1=deepcopy(activation),
                norm2=norm,
                activation2=deepcopy(activation),
            )
        )

        # atrous convs
        for dilation in dilations:
            self.convs.append(
                DepthwiseSeparableConv2d(
                    in_channels,
                    out_channels // 8,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    norm1=norm,
                    activation1=deepcopy(activation),
                    norm2=norm,
                    activation2=deepcopy(activation),
                )
            )

    def forward(self, x):
        res = []
        y = self.convs[0](x)
        res.append(y)
        x_1 = self.convs[1](x)
        res.append(x_1)
        x_2 = self.convs[2](x)
        res.append(x_2)
        x_3 = self.gap(x_1).expand_as(x_1)
        res.append(x_3)
        x_4 = self.gap(x_2).expand_as(x_2)
        res.append(x_4)
        res = torch.cat(res, dim=1)
        return res
