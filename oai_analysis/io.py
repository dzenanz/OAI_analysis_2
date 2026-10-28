"""
Reading inputs and writing outputs.
"""
from pathlib import Path

import itk
import numpy as np
import vtk

from oai_analysis.algorithms import mesh_processing as mp

SERIES_DESCRIPTION_TAG = "0008|103e"


def read_image(path) -> itk.Image:
    """Read an image file, or a DICOM series from a directory, with float pixels."""
    return itk.imread(path, pixel_type=itk.F)


def series_description(path, image=None) -> str | None:
    """The DICOM series description of the input, if it has one."""
    if Path(path).is_dir():  # DICOM series
        names_generator = itk.GDCMSeriesFileNames.New()
        names_generator.SetUseSeriesDetails(True)
        names_generator.SetDirectory(str(path))
        metadata = itk.imread(names_generator.GetInputFileNames()[0]).GetMetaDataDictionary()
    else:
        metadata = (image if image is not None else itk.imread(path)).GetMetaDataDictionary()
    if metadata.HasKey(SERIES_DESCRIPTION_TAG):
        return metadata[SERIES_DESCRIPTION_TAG]
    return None


def write(obj, path, compression=False):
    """Write an output, choosing the writer by the object's type. `compression` applies to images and transforms."""
    path = str(path)
    if isinstance(obj, bytes):
        Path(path).write_bytes(obj)
    elif isinstance(obj, np.ndarray):
        np.save(path, obj)
    elif isinstance(obj, vtk.vtkPolyData):
        mp.write_vtk_mesh(obj, path)
    elif isinstance(obj, itk.Image):
        itk.imwrite(obj, path, compression=compression)
    elif isinstance(obj, itk.Transform):
        # HDF5 cannot compress an empty parameter array, e.g. of an identity transform
        itk.transformwrite(obj, path, compression=compression and obj.GetNumberOfParameters() > 0)
    else:
        raise TypeError(f"Do not know how to write {type(obj).__name__} to {path}")


def default_extension(obj) -> str:
    """The extension used for an output of this type in the flat output layout."""
    if isinstance(obj, np.ndarray):
        return ".npy"
    if isinstance(obj, vtk.vtkPolyData):
        return ".vtk"
    if isinstance(obj, itk.Image):
        return ".nrrd"
    if isinstance(obj, itk.Transform):
        return ".tfm"
    raise TypeError(f"No default extension for {type(obj).__name__}")


IMAGE_FORMATS = (".nrrd", ".nii.gz")
TRANSFORM_FORMATS = (".h5", ".tfm")


def extension(obj, image_format=".nrrd", transform_format=".h5") -> str:
    """The extension for an output of this type, with the formats chosen for images and transforms."""
    if isinstance(obj, itk.Image):
        return image_format
    if isinstance(obj, itk.Transform):
        return transform_format
    return default_extension(obj)


def read(path):
    """Read an output written by write(), choosing the reader by the file's extension."""
    path = str(path)
    if path.endswith((".nrrd", ".nii.gz", ".nii")):
        return itk.imread(path)
    if path.endswith((".h5", ".tfm")):
        return itk.transformread(path)[0]
    if path.endswith(".vtk"):
        return mp.read_vtk_mesh(path)
    if path.endswith(".npy"):
        return np.load(path)
    return Path(path).read_bytes()
