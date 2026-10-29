"""
Small PNG pictures of a case, to look at without a viewer: the segmentation over the image, and the 2D thickness
maps. A project writes them to the case's thumbnails/ folder as soon as the step they show is done.
"""
import io

import numpy as np
from matplotlib.figure import Figure

from oai_analysis.algorithms.cartilage_shape_processing import thickness_grid

COLORS = {"FC": (1.0, 0.3, 0.2), "TC": (0.2, 0.6, 1.0)}
MAX_THICKNESS_MM = 5.0  # the top of the color scale, the same for every case so that they compare
MAP_RESOLUTION = 100


def segmentation(case) -> dict[str, bytes]:
    """The sagittal slice with the most cartilage, with the cartilage colored."""
    import itk

    image = itk.array_view_from_image(case.image)  # slice, row, column
    masks = {s: itk.array_view_from_image(p) > 0.5 for s, p in case.probability.items()}
    index = int(np.argmax(sum(mask.sum(axis=(1, 2)) for mask in masks.values())))
    rgb = np.repeat(_normalized(image[index])[..., None], 3, axis=2)
    for structure, mask in masks.items():
        rgb[mask[index]] = 0.5 * rgb[mask[index]] + 0.5 * np.array(COLORS.get(structure, (1.0, 1.0, 0.0)))
    spacing = case.image.GetSpacing()
    figure = Figure(figsize=(4, 4 * image.shape[1] * spacing[1] / (image.shape[2] * spacing[0])))
    axes = figure.add_axes((0, 0, 1, 1))
    axes.imshow(rgb, aspect=spacing[1] / spacing[0])
    axes.axis("off")
    axes.text(4, 4, f"{case.case.case_id}, slice {index}: " + ", ".join(
        f"{s} {'red' if s == 'FC' else 'blue'}" for s in masks), color="white", fontsize=7, va="top")
    return {"overlay.png": _png(figure)}


def thickness_maps(case) -> dict[str, bytes]:
    """The thickness on the atlas, over its 2D embedding, from 0 to MAX_THICKNESS_MM."""
    pictures = {}
    for structure in case.atlas_mapped:
        grid, extent = thickness_grid(case.atlas.embedding_2d[structure], case.atlas_thickness(structure).copy(),
                                      ninter=MAP_RESOLUTION)
        figure = Figure(figsize=(3, 3))
        axes = figure.add_subplot()
        shown = axes.imshow(np.flipud(grid), extent=extent, vmin=0, vmax=MAX_THICKNESS_MM)
        axes.axis("equal")
        axes.axis("off")
        figure.colorbar(shown, ax=axes, label="mm").ax.tick_params(labelsize=7)
        axes.set_title(f"{case.case.case_id} {structure}", fontsize=8)
        pictures[f"{structure}.png"] = _png(figure)
    return pictures


# step name -> what makes its thumbnails
BY_STEP = {"segment": segmentation, "atlas_map": thickness_maps}


def _normalized(array):
    low, high = np.percentile(array, (1, 99))
    return np.clip((array - low) / (high - low or 1), 0, 1)


def _png(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100)
    return buffer.getvalue()
