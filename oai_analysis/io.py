"""
Reading inputs and writing outputs.
"""
from pathlib import Path

import itk
import numpy as np
import vtk

from oai_analysis import mesh_processing as mp

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
    """Write an output, choosing the writer by the object's type. `compression` applies to images."""
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
        itk.transformwrite(obj, path)
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
