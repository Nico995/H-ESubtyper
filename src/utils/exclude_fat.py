import openslide
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import h5py
from PIL import Image
from tqdm import tqdm
from concurrent.futures import as_completed, ProcessPoolExecutor
import traceback
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--slides_root",
    type=str,
    required=True,
)
parser.add_argument(
    "--slides_ext",
    type=str,
    required=True,
)
parser.add_argument(
    "--feats_root",
    type=str,
    required=True,
)
parser.add_argument(
    "--grid_ds_level",
    type=int,
    default=64,
)
parser.add_argument(
    "--min_tiles_tissue",
    type=int,
    default=50,
)


args = parser.parse_args()
args.slides_root = Path(args.slides_root)
args.feats_root = Path(args.feats_root)

args.feats_out = args.feats_root.parent / (args.feats_root.stem + "filtered")
args.checks_out = args.feats_root.parent / "filtered_grids"

args.feats_out.mkdir(exist_ok=True)
args.checks_out.mkdir(exist_ok=True)


def stemmize(x):
    return np.array([xx.stem for xx in x])


def get_mag(slide):
    objective_power = slide.properties.get("aperio.AppMag")

    if objective_power is None:
        # Fallback options (vendor-dependent)
        objective_power = slide.properties.get("openslide.objective-power")

    return objective_power


def downsample(image, level, resample=Image.BILINEAR):
    if level < 0:
        raise ValueError("levels must be >= 1")
    elif level == 0:
        return image
    else:
        factor = 2**level
        new_size = (image.width // factor, image.height // factor)
        return image.resize(new_size, resample=resample)


def draw_grid(image, coords, size, ds_level=6, color=(0, 0, 255), thickness=1):
    if isinstance(image, Image.Image):
        image = np.array(image)

    ds = 2**ds_level

    # Downsample size and coordinates
    downsampled_size = (size[0] // ds, size[1] // ds)
    coords = [(x // ds, y // ds) for x, y in coords]

    for x, y in coords:
        pt1 = (x, y)
        pt2 = (x + downsampled_size[0], y + downsampled_size[1])
        cv2.rectangle(image, pt1, pt2, color, thickness)

    return image


def draw_white_margin(image, margin=25):
    image[:margin, :] = 255
    image[-margin:, :] = 255
    image[:, :margin] = 255
    image[:, -margin:] = 255

    return image


def get_adaptive_threshold_contours(
    image,
    ds_level,
    blur_kernel=(5, 5),
    block_size=11,
    C=2,
    margin=25,
    tile_size=512,
    min_tiles_tissue=50,
):

    ds = 2**ds_level
    tile_size_ds = tile_size // ds
    min_area = (tile_size_ds**2) * min_tiles_tissue

    image = draw_white_margin(image, margin)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    _, thresh = cv2.threshold(
        blurred, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # contours, _ = cv2.findContours(
    #     thresh,
    #     mode=cv2.RETR_EXTERNAL,
    #     method=cv2.CHAIN_APPROX_SIMPLE
    # )
    contours, hierarchy = cv2.findContours(
        thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )

    hierarchy = hierarchy[0]  # shape: (N, 4)

    # Keep only external contours (no parent)
    external_contours = [
        cnt
        for cnt, h in zip(contours, hierarchy)
        if h[3] == 0 and cv2.contourArea(cnt) >= min_area
    ]

    # Keep only holes (have parent)
    hole_contours = [
        cnt
        for cnt, h in zip(contours, hierarchy)
        if h[3] > 0 and cv2.contourArea(cnt) > 5
    ]

    return external_contours, hole_contours


def filter_coords_in_contours(
    coords, external_contours, hole_contours, ds_level, tile_size=512
):
    mask = []

    ds = 2**ds_level

    tile_size_ds = tile_size // ds

    for coord in coords:
        x0, y0 = coord[0] // ds, coord[1] // ds
        corners = [
            (float(x0), float(y0)),
            (float(x0 + tile_size_ds), float(y0)),
            (float(x0), float(y0 + tile_size_ds)),
            (float(x0 + tile_size_ds), float(y0 + tile_size_ds)),
        ]

        inside_any = False
        for pt in corners:
            in_outer = any(
                cv2.pointPolygonTest(cnt, pt, False) >= 0 for cnt in external_contours
            )
            in_hole = any(
                cv2.pointPolygonTest(cnt, pt, False) >= 0 for cnt in hole_contours
            )

            if in_outer and not in_hole:
                inside_any = True
                break

        mask.append(inside_any)

    return np.array(mask)


def process_slide(
    slide_path,
    feat_path,
    out_dir,
    ds=128,
    margin=25,
    tile_size=512,
    min_tiles_tissue=50,
):
    try:
        slide = openslide.OpenSlide(slide_path)

        # Get downsampling info
        slide_ds_factors = np.array(
            slide.level_downsamples
        )  # e.g. [1.0, 2.0, 4.1, 8.3, ...]
        best_level_idx = np.max(
            np.where(slide_ds_factors <= ds)[0]
        )  # closest level ≤ ds
        wsi_downsample_factor = slide_ds_factors[best_level_idx]
        manual_downsample_factor = ds / wsi_downsample_factor
        manual_downsample_level = int(round(np.log2(manual_downsample_factor)))
        true_ds_level = int(round(np.log2(ds)))  # used for contouring, grid, etc.

        # Read and rescale image
        image_to_contour = slide.read_region(
            (0, 0), best_level_idx, slide.level_dimensions[best_level_idx]
        ).convert("RGB")
        image_to_contour = downsample(image_to_contour, manual_downsample_level)
        image_to_contour = np.array(image_to_contour)

        # Load features
        with h5py.File(feat_path, "r") as f:
            coords = f["coords"][()]
            feats = f["features"][()]

        # Contours at ds=2^true_ds_level
        external_contours, hole_contours = get_adaptive_threshold_contours(
            image_to_contour,
            true_ds_level,
            margin=margin,
            tile_size=tile_size,
            min_tiles_tissue=min_tiles_tissue,
        )

        # Filter coordinates by contours
        inside_mask = filter_coords_in_contours(
            coords, external_contours, hole_contours, true_ds_level
        )

        coords_inside = coords[inside_mask]
        coords_outside = coords[~inside_mask]
        feats_inside = feats[inside_mask]

        # Draw results
        grid_image = draw_grid(
            image_to_contour,
            coords_inside,
            size=(tile_size, tile_size),
            ds_level=true_ds_level,
            thickness=1,
            color=(0, 255, 0),
        )
        grid_image = draw_grid(
            grid_image,
            coords_outside,
            size=(tile_size, tile_size),
            ds_level=true_ds_level,
            thickness=1,
            color=(255, 0, 0),
        )
        grid_image = cv2.drawContours(grid_image, hole_contours, -1, (255, 0, 0), 1)
        grid_image = cv2.drawContours(grid_image, external_contours, -1, (0, 255, 0), 1)

        # Write output
        with h5py.File(out_dir / feat_path.name, "w") as f_out:
            f_out.create_dataset("coords", data=coords_inside)
            f_out.create_dataset("features", data=feats_inside)

            with h5py.File(feat_path, "r") as f_in:
                for key, val in f_in.attrs.items():
                    f_out["coords"].attrs[key] = val

        return {
            "grid_image": grid_image,
            "inside_mask": inside_mask,
            "slide_path": slide_path,
            "feat_path": feat_path,
        }
    except Exception as e:
        return {
            "error": f"Error processing {slide_path}",
            "details": traceback.format_exc(),
        }


def process_slides_parallel(
    slides_paths,
    feats_paths,
    out_root,
    grid_ds_level,
    min_tiles_tissue,
):
    # Parallel execution
    results = []
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(
                process_slide,
                slide_path=slide_path,
                feat_path=feat_path,
                out_dir=out_root,
                ds=grid_ds_level,
                min_tiles_tissue=min_tiles_tissue,
            ): (
                slide_path,
                feat_path,
            )
            for slide_path, feat_path in zip(slides_paths, feats_paths)
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing slides..."
        ):
            result = future.result()
            results.append(result)

    # Extract only successful ones
    grids_images = [r["grid_image"] for r in results if "error" not in r]
    inside_masks = [r["inside_mask"] for r in results if "error" not in r]

    # Optional: print errors
    for r in results:
        if "error" in r:
            print(r["error"])

    names = [r["slide_path"].stem for r in results if "error" not in r]

    print("Saving grids...")
    for img, name in tqdm(zip(grids_images, names), total=len(names)):
        save_path = args.checks_out / (name + ".jpg")
        Image.fromarray(img).save(save_path)


if __name__ == "__main__":
    slides_paths = sorted(list(args.slides_root.glob(f"*.{args.slides_ext}")))
    feats_paths = sorted(list(args.feats_root.glob("*.h5")))

    slides = [slide.stem for slide in slides_paths]
    feats = [feat.stem for feat in feats_paths]

    commons = set(feats).intersection(slides)

    slides_paths = sorted(
        [args.slides_root / (common + "." + args.slides_ext) for common in commons]
    )
    feats_paths = sorted([args.feats_root / (common + ".h5") for common in commons])

    assert (
        stemmize(slides_paths) == stemmize(feats_paths)
    ).all(), "Mismatch between slides and features"

    process_slides_parallel(
        slides_paths,
        feats_paths,
        args.feats_out,
        args.grid_ds_level,
        args.min_tiles_tissue,
    )
