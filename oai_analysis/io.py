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
    # GetMetaDataDictionary() returns a reference into its owner, so the owner is kept in a variable while it is read
    if Path(path).is_dir():  # DICOM series: read the header of one file
        names_generator = itk.GDCMSeriesFileNames.New()
        names_generator.SetUseSeriesDetails(True)
        names_generator.SetDirectory(str(path))
        owner = itk.GDCMImageIO.New()
        owner.SetFileName(names_generator.GetInputFileNames()[0])
        owner.ReadImageInformation()
    else:
        owner = image if image is not None else itk.imread(path)
    metadata = owner.GetMetaDataDictionary()
    if metadata.HasKey(SERIES_DESCRIPTION_TAG):
        return metadata[SERIES_DESCRIPTION_TAG]
    return None


def size_and_acquisition(path) -> tuple[int, tuple[str, float] | None]:
    """
    The number of voxels of an image or a DICOM series, and when it was acquired as (date, time) if the DICOM
    header says. Reads one header, not the pixels.
    """
    path = Path(path)
    if path.is_dir():  # DICOM series, one slice per file
        files = sorted(f for f in path.iterdir() if f.is_file())
        image_io = itk.GDCMImageIO.New()
        image_io.SetFileName(str(files[0]))
        image_io.ReadImageInformation()
        voxels = image_io.GetDimensions(0) * image_io.GetDimensions(1) * len(files)
        metadata = image_io.GetMetaDataDictionary()
        for date_tag, time_tag in (("0008|0022", "0008|0032"), ("0008|0021", "0008|0031")):  # acquisition, series
            if metadata.HasKey(date_tag) and metadata.HasKey(time_tag):
                date, time = metadata[date_tag].strip(), metadata[time_tag].strip()
                if date and time:
                    return voxels, (date, float(time))
        return voxels, None
    image_io = itk.ImageIOFactory.CreateImageIO(str(path), itk.CommonEnums.IOFileMode_ReadMode)
    if image_io is None:
        raise OSError(f"cannot read {path}")
    image_io.SetFileName(str(path))
    image_io.ReadImageInformation()
    return int(np.prod([image_io.GetDimensions(d) for d in range(image_io.GetNumberOfDimensions())])), None


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
