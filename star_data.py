"""
FPGA Star Tracker - Dataset Generator
======================================
Generates synthetic starfield images for 6 orientation classes.

Class definitions (geometric rules):
  up      - >= 65% of stars in top half
  down    - >= 65% of stars in bottom half
  left    - >= 65% of stars in left half
  right   - >= 65% of stars in right half
  forward - stars spread outward from center (high radial spread)
  back    - stars clustered toward center (low radial spread)

Usage:
    python generate_dataset.py

Outputs:
    starfield_dataset/
        up/         0000.png ... N.png
        down/       ...
        left/       ...
        right/      ...
        forward/    ...
        back/       ...
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_W          = 320
IMG_H          = 240
IMAGES_PER_CLASS = 500        # increase for more data (recommended: 1000+)
DATASET_DIR    = "starfield_dataset"
CLASSES        = ["up", "down", "left", "right", "forward", "back"]

# Star appearance
MIN_STARS      = 40
MAX_STARS      = 120
STAR_RADIUS    = 1             # px radius (0 = single pixel, 1 = small dot)
STAR_BRIGHTNESS_MIN = 180      # grayscale 0-255
STAR_BRIGHTNESS_MAX = 255

# Class geometry thresholds
BIAS_RATIO     = 0.70          # fraction of stars in the "dominant" region
NOISE_RATIO    = 1 - BIAS_RATIO  # remaining stars placed randomly

# Forward/back radial parameters
FORWARD_MIN_RADIAL = 0.55      # mean radial dist as fraction of half-diagonal
FORWARD_MAX_RADIAL = 0.90
BACK_MIN_RADIAL    = 0.05
BACK_MAX_RADIAL    = 0.40


# ─────────────────────────────────────────────
# STAR PLACEMENT HELPERS
# ─────────────────────────────────────────────

def random_star(x_range, y_range):
    """Return (x, y) uniformly sampled from given ranges."""
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)
    return x, y


def random_star_full():
    return random_star((0, IMG_W), (0, IMG_H))


def radial_star(min_r_frac, max_r_frac):
    """
    Sample a star at a radial distance from image center.
    r is expressed as a fraction of the half-diagonal.
    Uses rejection sampling to stay within image bounds.
    """
    cx, cy = IMG_W / 2, IMG_H / 2
    half_diag = np.sqrt(cx**2 + cy**2)
    min_r = min_r_frac * half_diag
    max_r = max_r_frac * half_diag

    for _ in range(200):  # max attempts before giving up
        r = random.uniform(min_r, max_r)
        theta = random.uniform(0, 2 * np.pi)
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        if 0 <= x < IMG_W and 0 <= y < IMG_H:
            return x, y

    # Fallback: random position in outer ring area
    return random_star_full()


def place_stars(n_stars, biased_fn, noise_fn=random_star_full):
    """
    Place n_stars total:
      - BIAS_RATIO fraction via biased_fn  (class-specific region)
      - NOISE_RATIO fraction via noise_fn  (random anywhere)
    Returns list of (x, y) tuples.
    """
    n_biased = int(round(n_stars * BIAS_RATIO))
    n_noise  = n_stars - n_biased

    stars = [biased_fn() for _ in range(n_biased)]
    stars += [noise_fn() for _ in range(n_noise)]
    random.shuffle(stars)
    return stars


# ─────────────────────────────────────────────
# PER-CLASS STAR GENERATORS
# ─────────────────────────────────────────────

def stars_up(n):
    return place_stars(n,
        biased_fn=lambda: random_star((0, IMG_W), (0, IMG_H * 0.5)))

def stars_down(n):
    return place_stars(n,
        biased_fn=lambda: random_star((0, IMG_W), (IMG_H * 0.5, IMG_H)))

def stars_left(n):
    return place_stars(n,
        biased_fn=lambda: random_star((0, IMG_W * 0.5), (0, IMG_H)))

def stars_right(n):
    return place_stars(n,
        biased_fn=lambda: random_star((IMG_W * 0.5, IMG_W), (0, IMG_H)))

def stars_forward(n):
    """Stars spread toward edges — high radial distance from center."""
    return place_stars(n,
        biased_fn=lambda: radial_star(FORWARD_MIN_RADIAL, FORWARD_MAX_RADIAL))

def stars_back(n):
    """Stars clustered near center — low radial distance."""
    return place_stars(n,
        biased_fn=lambda: radial_star(BACK_MIN_RADIAL, BACK_MAX_RADIAL))


CLASS_GENERATORS = {
    "up":      stars_up,
    "down":    stars_down,
    "left":    stars_left,
    "right":   stars_right,
    "forward": stars_forward,
    "back":    stars_back,
}


# ─────────────────────────────────────────────
# IMAGE RENDERER
# ─────────────────────────────────────────────

def render_starfield(star_positions):
    """
    Render star positions onto a black background.
    Stars are white/gray dots with slight brightness variation.
    Returns a PIL Image (grayscale 'L').
    """
    img = Image.new("L", (IMG_W, IMG_H), color=0)
    draw = ImageDraw.Draw(img)

    for (x, y) in star_positions:
        brightness = random.randint(STAR_BRIGHTNESS_MIN, STAR_BRIGHTNESS_MAX)
        xi, yi = int(x), int(y)

        if STAR_RADIUS == 0:
            if 0 <= xi < IMG_W and 0 <= yi < IMG_H:
                img.putpixel((xi, yi), brightness)
        else:
            r = STAR_RADIUS
            draw.ellipse(
                [(xi - r, yi - r), (xi + r, yi + r)],
                fill=brightness
            )

    # Optional: very slight Gaussian-like blur to soften stars
    # from PIL import ImageFilter
    # img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    return img


# ─────────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────────

def generate_class(class_name, out_dir, n_images):
    os.makedirs(out_dir, exist_ok=True)
    generator = CLASS_GENERATORS[class_name]
    count = 0

    for i in range(n_images):
        n_stars = random.randint(MIN_STARS, MAX_STARS)
        star_positions = generator(n_stars)
        img = render_starfield(star_positions)
        fname = f"{i:04d}.png"
        img.save(os.path.join(out_dir, fname))
        count += 1

    return count


def generate_all():
    print("=" * 50)
    print("  STAR TRACKER DATASET GENERATOR")
    print("=" * 50)
    print(f"  Resolution : {IMG_W} x {IMG_H}")
    print(f"  Images/cls : {IMAGES_PER_CLASS}")
    print(f"  Bias ratio : {BIAS_RATIO:.0%} in dominant region")
    print(f"  Classes    : {', '.join(CLASSES)}")
    print()

    total = 0
    for class_name in CLASSES:
        out_dir = os.path.join(DATASET_DIR, class_name)
        print(f"  Generating '{class_name}' → {out_dir} ...", end=" ", flush=True)
        n = generate_class(class_name, out_dir, IMAGES_PER_CLASS)
        print(f"{n} images saved.")
        total += n

    print()
    print(f"  Done. Total images: {total}")
    print(f"  Dataset root: {os.path.abspath(DATASET_DIR)}")
    print("=" * 50)


if __name__ == "__main__":
    generate_all()