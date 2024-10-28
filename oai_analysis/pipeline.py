import os
import pathlib

import icon_registration.itk_wrapper as itk_wrapper
import itk
import vtk
from unigradicon import preprocess, get_unigradicon

import mesh_processing as mp
from analysis_object import AnalysisObject
from cartilage_shape_processing import thickness_3d_to_2d

DATA_DIR = pathlib.Path(__file__).parent / "data"


def write_vtk_mesh(mesh, filename):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(mesh)
    writer.SetFileVersion(42)  # ITK does not support newer version (5.1)
    writer.SetFileTypeToBinary()  # reading and writing binary files is faster
    writer.Write()


def transform_mesh(mesh, transform, filename_prefix, keep_intermediate_outputs):
    """
    Transform the input mesh using the provided transform.

    :param mesh: input mesh
    :param transform: The modelling transform to use (the inverse of image resampling transform)
    :param filename_prefix: prefix (including path) for the intermediate file names
    :return: transformed mesh
    """
    itk_mesh = mp.get_itk_mesh(mesh)
    t_mesh = itk.transform_mesh_filter(itk_mesh, transform=transform)
    # itk.meshwrite(t_mesh, filename_prefix + "_transformed.vtk", binary=True)  # does not work in 5.4 and earlier
    itk.meshwrite(t_mesh, filename_prefix + "_transformed.vtk", compression=True)

    transformed_mesh = mp.read_vtk_mesh(filename_prefix + "_transformed.vtk")

    if keep_intermediate_outputs:
        write_vtk_mesh(mesh, filename_prefix + "_original.vtk")
    else:
        os.remove(filename_prefix + "_transformed.vtk")

    return transformed_mesh


def into_canonical_orientation(image, flip_left_right):
    """
    Reorient the given image into the canonical orientation.

    :param image: input image
    :param flip_left_right: if True, flips the image left-right
    :return: reoriented image
    """
    dicom_lps = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RAI
    dicom_ras = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_LPI
    dicom_pir = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_ASL
    if flip_left_right:
        image_dir = itk.array_from_matrix(image.GetDirection())
        image_dir[0, :] *= -1  # flip left-right
        image.SetDirection(itk.matrix_from_array(image_dir))
    oriented_image = itk.orient_image_filter(
        image,
        use_image_direction=True,
        # given_coordinate_orientation=dicom_lps,
        # desired_coordinate_orientation=dicom_ras,
        desired_coordinate_orientation=dicom_pir,  # atlas' orientation
    )
    return oriented_image


def thickness_analysis(normalized_image, output_prefix=None):
    """
    Computes cartilage thickness for femur and tibia from knee MRI.

    :param normalized_image:
    :param output_prefix: If provided, writes intermediate results to files with this prefix.
    :return: Inner part of the mesh with distance information in attributes
    """
    print("Segmenting the femoral and tibial cartilage")
    obj = AnalysisObject()
    FC_prob, TC_prob = obj.segment(normalized_image)

    print("Computing the thickness map for the meshes")
    distance_inner_FC, distance_outer_FC = mp.get_thickness_mesh(FC_prob, mesh_type='FC')
    distance_inner_TC, distance_outer_TC = mp.get_thickness_mesh(TC_prob, mesh_type='TC')

    if output_prefix is not None:
        print("Saving intermediate results")
        itk.imwrite(FC_prob.astype(itk.F), output_prefix + "_FC_prob.nrrd", compression=True)
        itk.imwrite(TC_prob.astype(itk.F), output_prefix + "_TC_prob.nrrd", compression=True)
        write_vtk_mesh(distance_outer_FC, output_prefix + "_FC_outer.vtk")
        write_vtk_mesh(distance_outer_TC, output_prefix + "_TC_outer.vtk")

    return distance_inner_FC, distance_inner_TC


def analysis_pipeline(input_path, output_path, laterality, keep_intermediate_outputs):
    """
    Computes cartilage thickness for femur and tibia from knee MRI.

    :param input_path: path to the input image file, or path to input directory containing DICOM image series.
    :param output_path: path to the desired directory for outputs.
    """
    in_image = itk.imread(input_path, pixel_type=itk.F)
    if pathlib.Path(input_path).is_dir():  # DICOM series
        namesGenerator = itk.GDCMSeriesFileNames.New()
        namesGenerator.SetUseSeriesDetails(True)
        namesGenerator.SetDirectory(input_path)
        first_slice = itk.imread(namesGenerator.GetInputFileNames()[0])
        metadata = first_slice.GetMetaDataDictionary()
    else:
        metadata = in_image.GetMetaDataDictionary()
    if metadata.HasKey("0008|103e"):
        print(f"Laterality: {laterality}, Series description: {metadata['0008|103e']}")

    in_image = into_canonical_orientation(in_image, laterality == "right")  # simplifies mesh processing
    in_image = preprocess(in_image, modality="mri")
    os.makedirs(output_path, exist_ok=True)  # also holds intermediate results
    if keep_intermediate_outputs:
        itk.imwrite(in_image, os.path.join(output_path, "in_image.nrrd"))
    distance_inner_FC, distance_inner_TC = thickness_analysis(in_image, output_prefix=os.path.join(output_path, "in"))

    atlas_filename = DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas.nii.gz"
    atlas_image = itk.imread(atlas_filename, itk.F)

    inner_mesh_fc_atlas = mp.read_vtk_mesh(
        DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas_FC_inner_mesh_smooth.ply")
    inner_mesh_tc_atlas = mp.read_vtk_mesh(
        DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas_TC_inner_mesh_smooth.ply")

    print("Registering the input image to the atlas")
    model = get_unigradicon()
    in_image_D = in_image.astype(itk.D)
    atlas_image_D = atlas_image.astype(itk.D)
    phi_AB, phi_BA = itk_wrapper.register_pair(model, in_image_D, atlas_image_D, finetune_steps=None)
    if keep_intermediate_outputs:
        print("Saving registration results")
        itk.transformwrite(phi_AB, os.path.join(output_path, "resampling.tfm"))
        itk.transformwrite(phi_BA, os.path.join(output_path, "modelling.tfm"))

    print("Transforming the thickness measurements to the atlas space")
    # we use modelling transform, which is the inverse of the image resampling transform
    transformed_mesh_FC = transform_mesh(distance_inner_FC, transform=phi_BA, filename_prefix=output_path + "/FC",
                                         keep_intermediate_outputs=keep_intermediate_outputs)
    transformed_mesh_TC = transform_mesh(distance_inner_TC, transform=phi_BA, filename_prefix=output_path + "/TC",
                                         keep_intermediate_outputs=keep_intermediate_outputs)

    print("Mapping the thickness to the atlas mesh")
    mapped_mesh_fc = mp.map_attributes(transformed_mesh_FC, inner_mesh_fc_atlas)
    mapped_mesh_tc = mp.map_attributes(transformed_mesh_TC, inner_mesh_tc_atlas)
    if keep_intermediate_outputs:
        write_vtk_mesh(mapped_mesh_fc, output_path + "/mapped_mesh_fc.vtk")
        write_vtk_mesh(mapped_mesh_tc, output_path + "/mapped_mesh_tc.vtk")

    print("Projecting thickness to 2D")
    thickness_3d_to_2d(mapped_mesh_fc, mesh_type='FC', output_filename=output_path + '/thickness_FC')
    thickness_3d_to_2d(mapped_mesh_tc, mesh_type='TC', output_filename=output_path + '/thickness_TC')


if __name__ == "__main__":
    test_cases = {
        r"M:\Dev\Osteoarthritis\OAI_analysis_2\oai_analysis\data\test_data\colab_case\image_preprocessed.nii.gz": {
            "laterality": "left", "name": "test_case"},
        r"M:\Dev\Osteoarthritis\NAS\OAIBaselineImages\results\0.C.2\9000798\20040924\10249506": {
            "laterality": "left", "name": "ENROLLMENT"},
        r"M:\Dev\Osteoarthritis\NAS\OAIBaselineImages\results\0.C.2\9000798\20040924\10249512": {
            "laterality": "right", "name": "ENROLLMENT"},
        r"M:\Dev\Osteoarthritis\NAS\OAI12MonthImages\results\1.C.2\9000798\20051130\10612103": {
            "laterality": "left", "name": "12_MONTH"},
        r"M:\Dev\Osteoarthritis\NAS\OAI12MonthImages\results\1.C.2\9000798\20051130\10612109": {
            "laterality": "right", "name": "12_MONTH"},
        r"M:\Dev\Osteoarthritis\NAS\OAI18MonthImages\results\2.D.2\9000798\20060303\10876803": {
            "laterality": "right", "name": "18_MONTH"},
        r"M:\Dev\Osteoarthritis\NAS\OAI24MonthImages\results\3.C.2\9000798\20070112\11415506": {
            "laterality": "left", "name": "24_MONTH"},
        r"M:\Dev\Osteoarthritis\NAS\OAI24MonthImages\results\3.C.2\9000798\20070112\11415512": {
            "laterality": "right", "name": "24_MONTH"},
    }

    for case, case_info in test_cases.items():
        print(f"\nProcessing {case}")
        output_path = f"./OAI_results_2/{case_info['name']}_{case_info['laterality']}"
        analysis_pipeline(case, output_path, case_info['laterality'], keep_intermediate_outputs=True)
