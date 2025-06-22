# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch
import itertools

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionWorldModel
from ultralytics.utils import DEFAULT_CFG, RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel

from .val import RTDETRDataset, RTDETRValidator

def on_pretrain_routine_end(trainer):
    """Callback."""
    if RANK in {-1, 0}:
        # NOTE: for evaluation
        names = [name.split("/")[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
        de_parallel(trainer.ema.ema).set_classes(names, cache_clip_model=False)
    device = next(trainer.model.parameters()).device
    trainer.text_model, _ = trainer.clip.load("ViT-B/32", device=device)
    for p in trainer.text_model.parameters():
        p.requires_grad_(False)

# def on_pretrain_routine_end(trainer):
#     """Callback."""
#     if RANK in {-1, 0}:
#         # NOTE: for evaluation
#         names = [name.split("/")[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
#         de_parallel(trainer.ema.ema).set_classes(names, cache_clip_model=False)
#     device = next(trainer.model.parameters()).device
#     ckpt = torch.load(f"/bigdata/XJH/RTDETR-main/remoteclip/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-{'ViT-B-32'}.pt", map_location='cuda:0')
#     model = trainer.open_clip.create_model_and_transforms('ViT-B-32')[0]
#     model = model.to(device)
#     model.load_state_dict(ckpt)
#     trainer.text_model = model
#     for p in trainer.text_model.parameters():
#         p.requires_grad_(False)

# RTDETRTrainer 继承自 DetectionTrainer 继承自baseTrainer（里面有loss）
class RTDETRTrainer(DetectionTrainer):
    """
    Trainer class for the RT-DETR model developed by Baidu for real-time object detection. Extends the DetectionTrainer
    class for YOLO to adapt to the specific features and architecture of RT-DETR. This model leverages Vision
    Transformers and has capabilities like IoU-aware query selection and adaptable inference speed.

    Notes:
        - F.grid_sample used in RT-DETR does not support the `deterministic=True` argument.
        - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.

    Example:
        ```python
        from ultralytics.models.rtdetr.train import RTDETRTrainer

        args = dict(model='rtdetr-l.yaml', data='coco8.yaml', imgsz=640, epochs=3)
        trainer = RTDETRTrainer(overrides=args)
        trainer.train()
        ```
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a WorldTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

        # Import and assign clip
        try:
            import clip
        except ImportError:
            checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip
        self.clip = clip
        # try:
        #     import open_clip
        # except ImportError:
        #     checks.check_requirements('open_clip')
        #     import open_clip
        # self.open_clip = open_clip
        
        

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (dict, optional): Model configuration. Defaults to None.
            weights (str, optional): Path to pre-trained model weights. Defaults to None.
            verbose (bool): Verbose logging if True. Defaults to True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """
        model = RTDETRDetectionWorldModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        return model

    def build_dataset(self, img_path, mode='val', batch=None):
        """
        Build and return an RT-DETR dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size for rectangle training. Defaults to None.

        Returns:
            (RTDETRDataset): Dataset object for the specific mode.
        """
        return RTDETRDataset(img_path=img_path,
                             imgsz=self.args.imgsz,
                             batch_size=batch,
                             augment=mode == 'train',
                             hyp=self.args,
                             rect=False,
                             cache=self.args.cache or None,
                             prefix=colorstr(f'{mode}: '),
                             data=self.data)

    def get_validator(self):
        """
        Returns a DetectionValidator suitable for RT-DETR model validation.

        Returns:
            (RTDETRValidator): Validator object for model validation.
        """
        self.loss_names = 'giou_loss', 'cls_loss', 'l1_loss'
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images. Scales and converts the images to float format.

        Args:
            batch (dict): Dictionary containing a batch of images, bboxes, and labels.

        Returns:
            (dict): Preprocessed batch.
        """
        batch = super().preprocess_batch(batch)
        # NOTE: add text features
        # 在图片基础上增加文本特征
        texts = list(itertools.chain(*batch["texts"]))
        #text_token = self.open_clip.tokenize(texts).to(batch["img"].device)
        text_token = self.clip.tokenize(texts).to(batch["img"].device)
        txt_feats = self.text_model.encode_text(text_token).to(dtype=batch["img"].dtype)  # torch.float32
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        bs = len(batch['img'])
        batch_idx = batch['batch_idx']
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch['bboxes'][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch['cls'][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        return batch
