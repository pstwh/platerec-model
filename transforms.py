import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose(
    [
        A.Resize(224, 224),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.33),
        A.Perspective(scale=(0.01, 0.08), pad_val=(114, 114, 114), p=1.0),
        A.OpticalDistortion(
            distort_limit=0.05,
            shift_limit=0.00,
            border_mode=0,
            value=(114, 114, 114),
            p=1.0,
        ),
        A.GridDistortion(
            num_steps=5, distort_limit=0.2, border_mode=0, value=(114, 114, 114), p=1.0
        ),
        A.SafeRotate(limit=(-10, 10), border_mode=0, value=(114, 114, 114), p=0.99),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
