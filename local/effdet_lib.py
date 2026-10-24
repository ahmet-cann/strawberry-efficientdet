"""Data pipeline for the EfficientDet strawberry notebook (local run).

Kept in a separate module so that DataLoader worker processes can import it:
on Windows, workers are started with `spawn` and cannot use classes defined
inside a Jupyter notebook.
"""
import cv2
import numpy as np
import torch
import albumentations as A
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PAD_VALUE = 114


def read_image(path):
    """Read an image as RGB. Works with non-ASCII paths on Windows (cv2.imread does not)."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'Cannot read image: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_yolo_labels(path):
    rows = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    rows.append([int(float(p[0]))] + [float(v) for v in p[1:5]])
    except FileNotFoundError:
        pass
    return rows


def _with_fill(cls, fill, **kw):
    try:
        return cls(fill=fill, **kw)
    except TypeError:
        return cls(cval=fill, **kw)


def build_augment(mode):
    """Augmentation levels compared in experiment E3: none, photometric, flip, scale."""
    t = []
    if mode in ('photometric', 'flip', 'scale'):
        t += [A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
              A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, p=0.3)]
    if mode in ('flip', 'scale'):
        t += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
    if mode == 'scale':
        t += [_with_fill(A.Affine, PAD_VALUE, scale=(0.7, 1.3), border_mode=cv2.BORDER_CONSTANT, p=0.5)]
    if not t:
        return None
    return A.Compose(t, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))


def letterbox(img, size):
    """Resize the longest side to `size` and pad to a square; returns scale and padding."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top


class StrawberryDataset(Dataset):
    def __init__(self, files, img_size, augment=None):
        self.files, self.img_size, self.augment = files, img_size, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_path, lbl_path = self.files[i]
        img = read_image(img_path)
        h, w = img.shape[:2]
        boxes, labels = [], []
        for c, xc, yc, bw, bh in read_yolo_labels(lbl_path):
            x1, y1 = max(0., (xc - bw / 2) * w), max(0., (yc - bh / 2) * h)
            x2, y2 = min(w, (xc + bw / 2) * w), min(h, (yc + bh / 2) * h)
            if x2 - x1 > 1 and y2 - y1 > 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(c)
        if self.augment is not None:
            out = self.augment(image=img, bboxes=boxes, labels=labels)
            img, boxes, labels = out['image'], [list(b) for b in out['bboxes']], list(out['labels'])
        img, r, left, top = letterbox(img, self.img_size)
        b = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        b = b * r + np.array([left, top, left, top], dtype=np.float32)
        x = torch.from_numpy(((img.astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1))
        target = {'bbox_xyxy': torch.from_numpy(b),
                  'labels': torch.tensor([int(l) for l in labels], dtype=torch.int64)}
        meta = {'path': img_path, 'orig_hw': (h, w), 'scale': r, 'pad': (left, top)}
        return x, target, meta


def collate(batch, img_size):
    """Batch for effdet: boxes as yxyx, classes shifted by +1 (0 = background)."""
    imgs = torch.stack([b[0] for b in batch])
    n = max(1, max(len(b[1]['labels']) for b in batch))
    bbox = torch.full((len(batch), n, 4), -1.)
    cls = torch.full((len(batch), n), -1.)
    for i, (_, t, _) in enumerate(batch):
        k = len(t['labels'])
        if k:
            bbox[i, :k] = t['bbox_xyxy'][:, [1, 0, 3, 2]]
            cls[i, :k] = t['labels'].float() + 1
    effdet_target = {'bbox': bbox, 'cls': cls,
                     'img_scale': torch.ones(len(batch)),
                     'img_size': torch.tensor([[img_size, img_size]] * len(batch), dtype=torch.float32)}
    return imgs, effdet_target, [b[1] for b in batch], [b[2] for b in batch]


def worker_init(_):
    cv2.setNumThreads(0)
