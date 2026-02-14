import numpy as np
import cv2
from PIL import Image


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
    min_tiles_hole=5,
):

    ds = 2**ds_level
    tile_size_ds = tile_size // ds
    min_tissue_area = (tile_size_ds**2) * min_tiles_tissue
    min_hole_area = (tile_size_ds**2) * min_tiles_hole

    image = draw_white_margin(image, margin)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    _, thresh = cv2.threshold(
        blurred, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, hierarchy = cv2.findContours(
        thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )

    hierarchy = hierarchy[0]  # shape: (N, 4)

    # Keep only external contours (no parent)
    external_contours = [
        cnt
        for cnt, h in zip(contours, hierarchy)
        if h[3] == 0 and cv2.contourArea(cnt) >= min_tissue_area
    ]

    # Keep only holes (have parent)
    hole_contours = [
        cnt
        for cnt, h in zip(contours, hierarchy)
        if h[3] > 0 and cv2.contourArea(cnt) > min_hole_area
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


import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from shapely.validation import make_valid


def get_union_of_thumbnail_contour_intersections(tissue_conts, gdf, level):
    intersections = []
    scale_up = 2 ** (level)

    # Convert gdf.exterior to shapely LineStrings
    exterior_lines = [
        LineString(geom.exterior.coords)
        for geom in gdf.geometry
        if not geom.is_empty and geom.geom_type == "Polygon"
    ]

    for cnt in tissue_conts:
        # shape: (N, 1, 2) → (N, 2)
        cnt = cnt.squeeze(1)
        if cnt.shape[0] < 3:
            continue

        # Upscale to level 0 space
        cnt_scaled = cnt * scale_up
        cnt_poly = Polygon(cnt_scaled)

        if not cnt_poly.is_valid:
            cnt_poly = make_valid(cnt_poly)

        for ext in exterior_lines:
            inter = cnt_poly.intersection(ext)
            if not inter.is_empty:
                intersections.append(inter)

    return unary_union(intersections) if intersections else None


from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np
import cv2


def extract_area_intersection_contours(cv_contours, gdf, level):
    scale_up = 2 ** (level)
    intersections = []

    for cnt in cv_contours:
        cnt = cnt.squeeze(1)
        if cnt.shape[0] < 3:
            continue

        cnt_scaled = cnt * scale_up
        cnt_poly = Polygon(cnt_scaled)
        if not cnt_poly.is_valid:
            cnt_poly = cnt_poly.buffer(0)

        for geom in gdf.geometry:
            if not geom.is_valid:
                geom = geom.buffer(0)

            inter = cnt_poly.intersection(geom)
            if not inter.is_empty:
                intersections.append(inter)

    # Union all overlapping areas
    union_intersection = unary_union(intersections)
    return intersections


def draw_geom_on_thumbnail(geom, thumbnail, draw_scale, color=(0, 0, 255), thickness=2):
    def to_cv2_coords(g):
        return (np.array(g.coords) * draw_scale).astype(np.int32).reshape(-1, 1, 2)

    if geom.geom_type == "LineString":
        cv2.polylines(
            thumbnail,
            [to_cv2_coords(geom)],
            isClosed=False,
            color=color,
            thickness=thickness,
        )
    elif geom.geom_type == "MultiLineString":
        for g in geom.geoms:
            cv2.polylines(
                thumbnail,
                [to_cv2_coords(g)],
                isClosed=False,
                color=color,
                thickness=thickness,
            )


def shapely_to_cv2_contours(geom):
    contours = []

    if geom.is_empty:
        return contours

    if geom.geom_type == "Polygon":
        coords = np.array(geom.exterior.coords).astype(np.int32).reshape(-1, 1, 2)
        contours.append(coords)
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            coords = np.array(poly.exterior.coords).astype(np.int32).reshape(-1, 1, 2)
            contours.append(coords)

    return contours


def _polygons_only(geom):
    """Return a list of Polygon objects contained in `geom` (recurses into collections)."""
    if geom is None or geom.is_empty:
        return []
    gt = geom.geom_type
    if gt == "Polygon":
        return [geom]
    if gt == "MultiPolygon":
        return [p for p in geom.geoms if not p.is_empty]
    if gt == "GeometryCollection":
        out = []
        for g in geom.geoms:
            out.extend(_polygons_only(g))
        return out
    # LineString, Point, etc. → ignore
    return []


# https://github.com/mahmoodlab/TRIDENT/blob/2c071412fab6d399cc21502050ecf43bae4faec3/trident/IO.py#L764
def overlay_gdf_on_thumbnail(
    gdf_contours,
    thumbnail,
    scale,
    tissue_color=(0, 255, 0),
    hole_color=(255, 0, 0),
    thickness=10,
):
    # Step 1: create a white mask
    mask = np.ones(thumbnail.shape[:2], dtype=np.uint8) * 255  # white background

    for geom in gdf_contours.geometry:
        if geom.is_empty:
            continue

        polys = _polygons_only(geom)

        for poly in polys:

            # External: fill in black to preserve tissue (leave background white)
            exterior = (np.array(poly.exterior.coords) * scale).astype(np.int32)
            cv2.fillPoly(mask, [exterior], color=0)

            # Holes: fill them back in white
            for hole in poly.interiors:
                hole_coords = (np.array(hole.coords) * scale).astype(np.int32)
                cv2.fillPoly(mask, [hole_coords], color=255)

            # Optionally draw outlines on original image
            cv2.polylines(
                thumbnail,
                [exterior],
                isClosed=True,
                color=tissue_color,
                thickness=thickness,
            )
            for hole in poly.interiors:
                hole_coords = (np.array(hole.coords) * scale).astype(np.int32)
                cv2.polylines(
                    thumbnail,
                    [hole_coords],
                    isClosed=True,
                    color=hole_color,
                    thickness=thickness,
                )

    # Step 2: apply mask: everything white except tissue region
    thumbnail[mask == 255] = (255, 255, 255)
