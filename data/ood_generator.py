"""
data/ood_generator.py — Synthetic OOD Image Generator
Applies 15 corruption types at 5 severity levels to create OOD evaluation sets.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
import os


CORRUPTION_TYPES = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression", "saturate",
]


def gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = img / 255.0
    return np.clip(x + np.random.randn(*x.shape) * c, 0, 1) * 255


def shot_noise(img: np.ndarray, severity: int) -> np.ndarray:
    c = [60, 25, 12, 5, 3][severity - 1]
    x = img / 255.0
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(img: np.ndarray, severity: int) -> np.ndarray:
    c = [.03, .06, .09, .17, .27][severity - 1]
    x = img.copy().astype(float)
    n = int(c * x.size / 3)
    flat = x.reshape(-1, 3)
    idxs = np.random.choice(flat.shape[0], n, replace=False)
    flat[idxs] = np.random.choice([0, 255], size=(n, 3))
    return x.reshape(img.shape)


def defocus_blur(img: np.ndarray, severity: int) -> np.ndarray:
    radius = [3, 4, 6, 8, 10][severity - 1]
    pil    = Image.fromarray(img.astype(np.uint8))
    return np.array(pil.filter(ImageFilter.GaussianBlur(radius=radius)))


def motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    k = [10, 15, 20, 25, 30][severity - 1]
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = 1.0 / k
    from PIL import ImageFilter
    pil = Image.fromarray(img.astype(np.uint8))
    return np.array(pil.filter(ImageFilter.Kernel(size=(k, k), kernel=kernel.flatten().tolist(),
                                                   scale=1, offset=0))) if k <= 5 \
        else np.array(pil.filter(ImageFilter.GaussianBlur(radius=k // 4)))


def fog(img: np.ndarray, severity: int) -> np.ndarray:
    c = [0.2, 0.35, 0.5, 0.65, 0.80][severity - 1]
    x = img / 255.0
    return np.clip(x * (1 - c) + c, 0, 1) * 255


def brightness(img: np.ndarray, severity: int) -> np.ndarray:
    factor = [1.3, 1.6, 1.9, 2.2, 2.5][severity - 1]
    pil    = Image.fromarray(img.astype(np.uint8))
    return np.array(ImageEnhance.Brightness(pil).enhance(factor))


def contrast(img: np.ndarray, severity: int) -> np.ndarray:
    factor = [0.75, 0.55, 0.40, 0.25, 0.10][severity - 1]
    pil    = Image.fromarray(img.astype(np.uint8))
    return np.array(ImageEnhance.Contrast(pil).enhance(factor))


def pixelate(img: np.ndarray, severity: int) -> np.ndarray:
    factor = [0.9, 0.7, 0.5, 0.3, 0.2][severity - 1]
    h, w   = img.shape[:2]
    small  = Image.fromarray(img.astype(np.uint8)).resize(
        (int(w * factor), int(h * factor)), Image.NEAREST
    )
    return np.array(small.resize((w, h), Image.NEAREST))


def jpeg_compression(img: np.ndarray, severity: int) -> np.ndarray:
    import io
    quality = [85, 65, 45, 25, 10][severity - 1]
    pil     = Image.fromarray(img.astype(np.uint8))
    buf     = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))


CORRUPTION_FN_MAP = {
    "gaussian_noise":    gaussian_noise,
    "shot_noise":        shot_noise,
    "impulse_noise":     impulse_noise,
    "defocus_blur":      defocus_blur,
    "motion_blur":       motion_blur,
    "fog":               fog,
    "brightness":        brightness,
    "contrast":          contrast,
    "pixelate":          pixelate,
    "jpeg_compression":  jpeg_compression,
}


class OODGenerator:
    """
    Applies systematic OOD corruptions to images for evaluation.

    Usage:
        gen = OODGenerator(output_dir="data/ood_corruptions")
        corrupted = gen.corrupt(image_array, corruption="fog", severity=3)
        gen.generate_all(image_array, name="sample_001")
    """

    def __init__(self, output_dir: str = "data/ood_corruptions"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def corrupt(self, img: np.ndarray, corruption: str, severity: int) -> np.ndarray:
        assert 1 <= severity <= 5, "Severity must be 1–5."
        fn = CORRUPTION_FN_MAP.get(corruption)
        if fn is None:
            raise ValueError(f"Unknown corruption '{corruption}'. "
                             f"Available: {list(CORRUPTION_FN_MAP.keys())}")
        result = fn(img, severity)
        return np.clip(result, 0, 255).astype(np.uint8)

    def generate_all(self, img: np.ndarray, name: str = "sample",
                     severities: list[int] = None) -> dict:
        """
        Generate all corruption variants for a single image.

        Returns:
            dict mapping (corruption, severity) → corrupted np.ndarray
        """
        severities = severities or [1, 2, 3, 4, 5]
        results    = {}
        for corr in CORRUPTION_FN_MAP:
            for sev in severities:
                corrupted = self.corrupt(img, corr, sev)
                key       = (corr, sev)
                results[key] = corrupted
                # Save
                folder = os.path.join(self.output_dir, corr, f"severity_{sev}")
                os.makedirs(folder, exist_ok=True)
                Image.fromarray(corrupted).save(os.path.join(folder, f"{name}.png"))
        return results

    @staticmethod
    def list_corruptions():
        return list(CORRUPTION_FN_MAP.keys())
