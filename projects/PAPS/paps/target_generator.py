import numpy as np
import torch
from amodal_panoptic_eval.labels.asd_labels import id2label


class PAPSAmodalTargetGenerator:
    """
    Generates training targets for PAPS.
    """

    def __init__(
        self,
        ignore_label,
        thing_ids,
        sigma=8,
        ignore_stuff_in_offset=False,
        small_instance_area=0,
        small_instance_weight=1,
        ignore_crowd_in_semantic=False,
    ):
        """
        Args:
            ignore_label: Integer, the ignore label for semantic segmentation.
            thing_ids: Set, a set of ids from contiguous category ids belonging
                to thing categories.
            sigma: the sigma for Gaussian kernel.
            ignore_stuff_in_offset: Boolean, whether to ignore stuff region when
                training the offset branch.
            small_instance_area: Integer, indicates largest area for small instances.
            small_instance_weight: Integer, indicates semantic loss weights for
                small instances.
            ignore_crowd_in_semantic: Boolean, whether to ignore crowd region in
                semantic segmentation branch, crowd region is ignored in the original
                TensorFlow implementation.
        """
        self.ignore_label = ignore_label
        self.thing_ids = set(thing_ids)
        self.ignore_stuff_in_offset = ignore_stuff_in_offset
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight
        self.ignore_crowd_in_semantic = ignore_crowd_in_semantic

        # Generate the default Gaussian image for each center
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    def __call__(self, panoptic, occ_level_gts):
        """
        Generate amodal target tensors for the given panoptic segmentation and occlusion level ground truths.

        Args:
            panoptic (ndarray): The panoptic segmentation array.
            occ_level_gts (list): A list of occlusion level amodal ground truth arrays.

        Returns:
            dict: A dictionary containing the following target tensors:
                - sem_seg: Semantic segmentation tensor.
                - center: Center heatmap tensor.
                - center_points: List of center points.
                - offset: Offset tensor.
                - sem_seg_weights: Semantic segmentation weights tensor.
                - center_weights: Center heatmap weights tensor.
                - offset_weights: Offset weights tensor.
                - thing_seg: Thing semantic segmentation tensor.
                - amodal_offsets: A dictionary of amodal offsets tensors for each occlusion level.
                - amodal_center_offsets: A dictionary of amodal center offsets tensors for each occlusion level.
                - amodal_offset_weights: A tensor of amodal offset weights.
                - occ_level_segs: A dictionary of occlusion level segmentation tensors.

        """

        height, width = panoptic.shape[0], panoptic.shape[1]
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label
        center = np.zeros((height, width), dtype=np.float32)
        center_pts = []
        amodal_offsets = []
        amodal_center_offsets = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord, x_coord = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        # Generate pixel-wise loss weights
        semantic_weights = np.ones_like(panoptic, dtype=np.uint8)
        # 0: ignore, 1: has instance
        # three conditions for a region to be ignored for instance branches:
        # (1) It is labeled as `ignore_label`
        # (2) It is crowd region (iscrowd=1)
        # (3) (Optional) It is stuff region (for offset branch)
        center_weights = np.zeros_like(panoptic, dtype=np.uint8)
        offset_weights = np.zeros_like(panoptic, dtype=np.uint8)
        amodal_offset_weights = np.zeros_like(panoptic, dtype=np.uint8)
        amodal_level_offsets = {}
        amodal_level_center_offsets = {}
        occ_level_segs = {}

        for occ_level_i, occ_level_gt in enumerate(occ_level_gts):
            occ_level_seg = np.zeros_like(panoptic, dtype=np.uint8)
            if occ_level_i != 0:
                amodal_offsets = np.zeros((2, height, width), dtype=np.float32)
                amodal_center_offsets = np.zeros((2, height, width), dtype=np.float32)
            else:
                amodal_offsets = None
                amodal_center_offsets = None

            occ_level_seg[occ_level_gt > 0] = 1
            occ_level_segs["occ_level_{}".format(occ_level_i)] = torch.as_tensor(
                occ_level_seg
            )
            inst_ids = np.unique(occ_level_gt)
            for inst_id in inst_ids:
                if inst_id == 0:
                    continue
                cat_id = inst_id // 1000
                semantic[panoptic == inst_id] = id2label[cat_id].trainId
                center_weights[panoptic == inst_id] = 1
                if id2label[cat_id].trainId in self.thing_ids:
                    offset_weights[panoptic == inst_id] = 1
                    amodal_offset_weights[occ_level_gt == inst_id] = 1
                    mask_index = np.where(panoptic == inst_id)
                    ins_area = len(mask_index[0])

                    if ins_area < self.small_instance_area:
                        semantic_weights[panoptic == inst_id] = (
                            self.small_instance_weight
                        )

                    center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                    center_pts.append([center_y, center_x])

                    y, x = int(round(center_y)), int(round(center_x))
                    sigma = self.sigma
                    # upper left
                    ul = int(np.round(x - 3 * sigma - 1)), int(
                        np.round(y - 3 * sigma - 1)
                    )
                    # bottom right
                    br = int(np.round(x + 3 * sigma + 2)), int(
                        np.round(y + 3 * sigma + 2)
                    )

                    # start and end indices in default Gaussian image
                    gaussian_x0, gaussian_x1 = max(0, -ul[0]), min(br[0], width) - ul[0]
                    gaussian_y0, gaussian_y1 = (
                        max(0, -ul[1]),
                        min(br[1], height) - ul[1],
                    )

                    # start and end indices in center heatmap image
                    center_x0, center_x1 = max(0, ul[0]), min(br[0], width)
                    center_y0, center_y1 = max(0, ul[1]), min(br[1], height)
                    center[center_y0:center_y1, center_x0:center_x1] = np.maximum(
                        center[center_y0:center_y1, center_x0:center_x1],
                        self.g[gaussian_y0:gaussian_y1, gaussian_x0:gaussian_x1],
                    )

                    # generate offset (2, h, w) -> (y-dir, x-dir)
                    offset[0][mask_index] = center_y - y_coord[mask_index]
                    offset[1][mask_index] = center_x - x_coord[mask_index]
                    if occ_level_i != 0:
                        amodal_mask_index = np.where(occ_level_gt == inst_id)
                        amodal_center_y, amodal_center_x = np.mean(
                            amodal_mask_index[0]
                        ), np.mean(amodal_mask_index[1])
                        amodal_center_offsets[
                            0, center_y0:center_y1, center_x0:center_x1
                        ] = (
                            amodal_center_y
                            - y_coord[center_y0:center_y1, center_x0:center_x1]
                        )
                        amodal_center_offsets[
                            1, center_y0:center_y1, center_x0:center_x1
                        ] = (
                            amodal_center_x
                            - x_coord[center_y0:center_y1, center_x0:center_x1]
                        )
                        amodal_offsets[0][amodal_mask_index] = (
                            amodal_center_y - y_coord[amodal_mask_index]
                        )
                        amodal_offsets[1][amodal_mask_index] = (
                            amodal_center_x - x_coord[amodal_mask_index]
                        )

            if occ_level_i != 0:
                amodal_level_offsets["occ_level_{}".format(occ_level_i)] = (
                    torch.as_tensor(amodal_offsets.astype(np.float32))
                )
                amodal_level_center_offsets["occ_level_{}".format(occ_level_i)] = (
                    torch.as_tensor(amodal_center_offsets.astype(np.float32))
                )

        center_weights = center_weights[None]
        offset_weights = offset_weights[None]
        amodal_offset_weights = amodal_offset_weights[None]
        thing_semantic = semantic.copy()
        mask_void = thing_semantic == self.ignore_label
        thing_semantic = thing_semantic - 10  # hard coded for now
        thing_semantic[mask_void] = 0
        stuff_ids = np.unique(panoptic[panoptic < 1000])
        for id_i in stuff_ids:
            if id_i == 0:
                continue
            semantic[panoptic == id_i] = id2label[id_i].trainId

        return dict(
            sem_seg=torch.as_tensor(semantic.astype("long")),
            center=torch.as_tensor(center.astype(np.float32)),
            center_points=center_pts,
            offset=torch.as_tensor(offset.astype(np.float32)),
            sem_seg_weights=torch.as_tensor(semantic_weights.astype(np.float32)),
            center_weights=torch.as_tensor(center_weights.astype(np.float32)),
            offset_weights=torch.as_tensor(offset_weights.astype(np.float32)),
            thing_seg=torch.as_tensor(thing_semantic.astype("long")),
            amodal_offsets=amodal_level_offsets,
            amodal_center_offsets=amodal_level_center_offsets,
            amodal_offset_weights=torch.as_tensor(
                amodal_offset_weights.astype(np.float32)
            ),
            occ_level_segs=occ_level_segs,
        )


class PAPSTargetGenerator:
    """
    Generates training targets for PAPS.
    """

    def __init__(
        self,
        ignore_label,
        thing_ids,
        sigma=8,
        ignore_stuff_in_offset=False,
        small_instance_area=0,
        small_instance_weight=1,
        ignore_crowd_in_semantic=False,
    ):
        """
        Args:
            ignore_label: Integer, the ignore label for semantic segmentation.
            thing_ids: Set, a set of ids from contiguous category ids belonging
                to thing categories.
            sigma: the sigma for Gaussian kernel.
            ignore_stuff_in_offset: Boolean, whether to ignore stuff region when
                training the offset branch.
            small_instance_area: Integer, indicates largest area for small instances.
            small_instance_weight: Integer, indicates semantic loss weights for
                small instances.
            ignore_crowd_in_semantic: Boolean, whether to ignore crowd region in
                semantic segmentation branch, crowd region is ignored in the original
                TensorFlow implementation.
        """
        self.ignore_label = ignore_label
        self.thing_ids = set(thing_ids)
        self.ignore_stuff_in_offset = ignore_stuff_in_offset
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight
        self.ignore_crowd_in_semantic = ignore_crowd_in_semantic

        # Generate the default Gaussian image for each center
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    def __call__(self, panoptic, segments_info):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py  # noqa
        reference: https://github.com/facebookresearch/detectron2/blob/main/datasets/prepare_panoptic_fpn.py#L18  # noqa

        Args:
            panoptic: numpy.array, panoptic label, we assume it is already
                converted from rgb image by panopticapi.utils.rgb2id.
            segments_info (list[dict]): see detectron2 documentation of "Use Custom Datasets".

        Returns:
            A dictionary with fields:
                - sem_seg: Tensor, semantic label, shape=(H, W).
                - center: Tensor, center heatmap, shape=(H, W).
                - center_points: List, center coordinates, with tuple
                    (y-coord, x-coord).
                - offset: Tensor, offset, shape=(2, H, W), first dim is
                    (offset_y, offset_x).
                - sem_seg_weights: Tensor, loss weight for semantic prediction,
                    shape=(H, W).
                - center_weights: Tensor, ignore region of center prediction,
                    shape=(H, W), used as weights for center regression 0 is
                    ignore, 1 is has instance. Multiply this mask to loss.
                - offset_weights: Tensor, ignore region of offset prediction,
                    shape=(H, W), used as weights for offset regression 0 is
                    ignore, 1 is has instance. Multiply this mask to loss.
        """
        height, width = panoptic.shape[0], panoptic.shape[1]
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label
        center = np.zeros((height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord, x_coord = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        # Generate pixel-wise loss weights
        semantic_weights = np.ones_like(panoptic, dtype=np.uint8)
        # 0: ignore, 1: has instance
        # three conditions for a region to be ignored for instance branches:
        # (1) It is labeled as `ignore_label`
        # (2) It is crowd region (iscrowd=1)
        # (3) (Optional) It is stuff region (for offset branch)
        center_weights = np.zeros_like(panoptic, dtype=np.uint8)
        offset_weights = np.zeros_like(panoptic, dtype=np.uint8)
        for seg in segments_info:
            cat_id = seg["category_id"]
            if not (self.ignore_crowd_in_semantic and seg["iscrowd"]):
                semantic[panoptic == seg["id"]] = cat_id
            if not seg["iscrowd"]:
                # Ignored regions are not in `segments_info`.
                # Handle crowd region.
                center_weights[panoptic == seg["id"]] = 1
                if not self.ignore_stuff_in_offset or cat_id in self.thing_ids:
                    offset_weights[panoptic == seg["id"]] = 1
            if cat_id in self.thing_ids:
                # find instance center
                mask_index = np.where(panoptic == seg["id"])
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[panoptic == seg["id"]] = self.small_instance_weight

                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])

                # generate center heatmap
                y, x = int(round(center_y)), int(round(center_x))
                sigma = self.sigma
                # upper left
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                # bottom right
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                # start and end indices in default Gaussian image
                gaussian_x0, gaussian_x1 = max(0, -ul[0]), min(br[0], width) - ul[0]
                gaussian_y0, gaussian_y1 = max(0, -ul[1]), min(br[1], height) - ul[1]

                # start and end indices in center heatmap image
                center_x0, center_x1 = max(0, ul[0]), min(br[0], width)
                center_y0, center_y1 = max(0, ul[1]), min(br[1], height)
                center[center_y0:center_y1, center_x0:center_x1] = np.maximum(
                    center[center_y0:center_y1, center_x0:center_x1],
                    self.g[gaussian_y0:gaussian_y1, gaussian_x0:gaussian_x1],
                )

                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset[0][mask_index] = center_y - y_coord[mask_index]
                offset[1][mask_index] = center_x - x_coord[mask_index]

        center_weights = center_weights[None]
        offset_weights = offset_weights[None]
        return dict(
            sem_seg=torch.as_tensor(semantic.astype("long")),
            center=torch.as_tensor(center.astype(np.float32)),
            center_points=center_pts,
            offset=torch.as_tensor(offset.astype(np.float32)),
            sem_seg_weights=torch.as_tensor(semantic_weights.astype(np.float32)),
            center_weights=torch.as_tensor(center_weights.astype(np.float32)),
            offset_weights=torch.as_tensor(offset_weights.astype(np.float32)),
        )
