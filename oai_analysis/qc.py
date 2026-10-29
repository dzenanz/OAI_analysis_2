"""
Measurements for the sanity checks of the default stages (see Stage.check). Each takes what a stage produced and
returns plain numbers; the stages compare them with their thresholds.
"""
import itk
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

STRUCTURE_NAMES = {"FC": "femoral cartilage", "TC": "tibial cartilage"}
# how many separate pieces a healthy segmentation of each structure has: the tibial cartilage has two plates
EXPECTED_PIECES = {"FC": 1, "TC": 2}


def finding(flagged: bool, code: str, message: str, value: float, severity="review"):
    """A finding with the given severity when flagged, otherwise an "info" one that only records the value."""
    from oai_analysis.stages.base import QCFinding

    return QCFinding(severity if flagged else "info", code, message, round(float(value), 6))


def percent(fraction) -> str:
    return f"{100 * fraction:.1f}%"


def axis_directions(image) -> tuple[int, ...]:
    """For each image axis, the physical axis closest to it, signed and counted from 1, e.g. (2, -3, 1)."""
    direction = itk.array_from_matrix(image.GetDirection())
    closest = np.argmax(np.abs(direction), axis=0)
    return tuple(int(axis + 1) * int(np.sign(direction[axis, i])) for i, axis in enumerate(closest))


def segmentation_measures(probability, structure) -> dict[str, float]:
    """
    volume_ml: of the voxels with probability above 0.5.
    outside_main_pieces: the fraction of that volume outside the structure's largest EXPECTED_PIECES pieces.
    uncertainty: the mean of 1 - |2p - 1| over the voxels with probability above 0.5: 0 when the network is sure.
    """
    p = itk.array_view_from_image(probability)
    mask = p > 0.5
    count = int(mask.sum())
    if count == 0:
        return {"volume_ml": 0.0, "outside_main_pieces": 0.0, "uncertainty": 1.0}
    labels = itk.array_view_from_image(itk.connected_component_image_filter(itk.image_from_array(mask.astype(np.uint8))))
    sizes = np.sort(np.bincount(labels.ravel())[1:])[::-1]
    return {"volume_ml": count * float(np.prod(probability.GetSpacing())) / 1000,
            "outside_main_pieces": float(sizes[EXPECTED_PIECES[structure]:].sum() / count),
            "uncertainty": float(np.mean(1 - np.abs(2 * p[mask] - 1)))}


def folding_fraction(transform) -> float:
    """The fraction of a registration's displacement field where it folds space (Jacobian determinant <= 0)."""
    fields = [itk.DisplacementFieldTransform[itk.D, 3].cast(transform.GetNthTransform(i)).GetDisplacementField()
              for i in range(transform.GetNumberOfTransforms())
              if transform.GetNthTransform(i).GetNameOfClass() == "DisplacementFieldTransform"]
    if not fields:
        return 0.0
    return float(np.mean([np.mean(jacobian_determinant(field) <= 0) for field in fields]))


def jacobian_determinant(field) -> np.ndarray:
    """det(I + grad u) at each voxel of a displacement field u, from central differences in physical units."""
    u = itk.array_from_image(field).astype(np.float64)  # z, y, x, then the x, y, z components
    direction = itk.array_from_matrix(field.GetDirection())
    sx, sy, sz = field.GetSpacing()
    jacobian = np.empty(u.shape[:3] + (3, 3))
    for c in range(3):
        dz, dy, dx = np.gradient(u[..., c], sz, sy, sx)
        # derivatives along the index axes, turned into derivatives along the physical axes
        jacobian[..., c, :] = np.stack([dx, dy, dz], axis=-1) @ direction.T
    jacobian += np.eye(3)
    return np.linalg.det(jacobian)


def similarity_to_atlas(image, atlas_image, phi_AB) -> float:
    """The correlation of the image, warped into atlas space, with the atlas image: 1 is a perfect match."""
    warped = itk.resample_image_filter(image.astype(itk.F), transform=phi_AB, use_reference_image=True,
                                       reference_image=atlas_image.astype(itk.F))
    return float(np.corrcoef(itk.array_view_from_image(warped).ravel(),
                             itk.array_view_from_image(atlas_image).ravel().astype(np.float32))[0, 1])


def point_values(mesh) -> np.ndarray:
    """The first per-vertex array of a mesh, e.g. its thickness."""
    array = mesh.GetPointData().GetArray(0)
    return vtk_to_numpy(array) if array is not None else np.zeros(0)


def mesh_is_empty(mesh: vtk.vtkPolyData) -> bool:
    return mesh is None or mesh.GetNumberOfPoints() == 0 or mesh.GetNumberOfCells() == 0
