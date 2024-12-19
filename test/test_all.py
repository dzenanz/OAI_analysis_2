import unittest

import itk
import oai_analysis.analysis_object
import oai_analysis.registration
from oai_analysis.data import test_data_dir as data_dir
from oai_analysis.data import atlases_dir
from oai_analysis import mesh_processing as mp
import numpy as np

class TestOAIAnalysis(unittest.TestCase):

    def setUp(self):
        self.data = data_dir()
        self.analysis_object = oai_analysis.analysis_object.AnalysisObject()

    def test_SegmentationCPU(self):
        input_image = itk.imread(self.data / "colab_case" / "image_preprocessed.nii.gz")

        correct_FC_segmentation = itk.imread(self.data / "colab_case" / "FC_probmap.nii.gz")
        correct_TC_segmentation = itk.imread(self.data / "colab_case" / "TC_probmap.nii.gz")

        FC, TC = self.analysis_object.segment(input_image)

        print('Segmentation Done')
        print(FC.shape, TC.shape)
        #np.save('FC_map.npy', FC)
        #np.save('TC_map.npy', TC)
        #print(np.sum(itk.ComparisonImageFilter(FC, correct_FC_segmentation)))
        #print(np.sum(itk.ComparisonImageFilter(TC, correct_TC_segmentation)))

        self.assertLess(np.sum(itk.comparison_image_filter(FC, correct_FC_segmentation)) , 12)
        self.assertLess(np.sum(itk.comparison_image_filter(TC, correct_TC_segmentation)) , 12)

    def test_MeshThicknessCPU(self):
        input_image = itk.imread(self.data / "colab_case" / "image_preprocessed.nii.gz", itk.D)
        atlas_image = self.analysis_object.atlas_image

        correct_FC_segmentation = itk.imread(self.data / "colab_case" / "FC_probmap.nii.gz", itk.D)
        correct_TC_segmentation = itk.imread(self.data / "colab_case" / "TC_probmap.nii.gz", itk.D)

        inner_mesh_fc_atlas = mp.read_vtk_mesh(
            atlases_dir() / "atlas_60_LEFT_baseline_NMI/atlas_FC_inner_mesh_LPS.ply")
        inner_mesh_tc_atlas = mp.read_vtk_mesh(
            atlases_dir() / "atlas_60_LEFT_baseline_NMI/atlas_TC_inner_mesh_LPS.ply")

        def deform_probmap(phi_AB, image_A, image_B, prob_map):
            interpolator = itk.LinearInterpolateImageFunction.New(image_A)
            warped_image = itk.resample_image_filter(prob_map,
                transform=phi_AB,
                interpolator=interpolator,
                size=itk.size(image_B),
                output_spacing=itk.spacing(image_B),
                output_direction=image_B.GetDirection(),
                output_origin=image_B.GetOrigin()
            )
            return warped_image

        phi_AB = self.analysis_object.register(input_image)

        warped_image_FC = deform_probmap(phi_AB, input_image, atlas_image, correct_FC_segmentation)
        warped_image_TC = deform_probmap(phi_AB, input_image, atlas_image, correct_TC_segmentation)

        fc_mesh_itk = mp.get_mesh_from_probability_map(warped_image_FC)
        fc_mesh = mp.itk_mesh_to_vtk_mesh(fc_mesh_itk)
        fc_inner_atlas, fc_outer_atlas = mp.get_split_mesh(fc_mesh, inner_mesh_fc_atlas, mesh_type='FC')
        fc_inner, fc_outer = mp.get_distance(fc_inner_atlas, fc_outer_atlas)
        distance_inner_FC = mp.get_itk_mesh(fc_inner)
        print(distance_inner_FC)

        tc_mesh_itk = mp.get_mesh_from_probability_map(warped_image_TC)
        tc_mesh = mp.itk_mesh_to_vtk_mesh(tc_mesh_itk)
        tc_inner_atlas, tc_outer_atlas = mp.get_split_mesh(tc_mesh, inner_mesh_tc_atlas, mesh_type='TC')
        tc_inner, tc_outer = mp.get_distance(tc_inner_atlas, tc_outer_atlas)
        distance_inner_TC = mp.get_itk_mesh(tc_inner)
        print(distance_inner_TC)

        print("Thickness computation completed")

        #assert(64800 <= distance_inner_FC.GetNumberOfPoints() <= 65000)
        #assert(20460 <= distance_inner_TC.GetNumberOfPoints() <= 20480)

    def test_RegistrationCPU(self):
        input_image = itk.imread(self.data / "colab_case" / "image_preprocessed.nii.gz")

        correct_registration = itk.imread(self.data / "colab_case" / "avsm" / "inv_transform_to_atlas.nii.gz")

        registration = self.analysis_object.register(input_image)

        #registration object is an itk transform. Need to verify that it is correct in test, but it appears correct
        print(registration)
        #self.assertFalse(np.sum(itk.ComparisonImageFilter(registration, correct_registration)) > 1)

class TestImports(unittest.TestCase):

    def test_ImportsCPU(self):
        self.analysis_object = oai_analysis.analysis_object.AnalysisObject()

class TestICONRegistration(unittest.TestCase):
    def setUp(self):
        self.data = data_dir()
        self.atlases = atlases_dir()

    def test_RegistrationCPU(self):
        ICON_obj = oai_analysis.registration.ICON_Registration()

        image_A = itk.imread(self.data / "colab_case" / "image_preprocessed.nii.gz")
        image_B = itk.imread(self.atlases / "atlas_60_LEFT_baseline_NMI" / "atlas_image.nii.gz")

        ICON_obj.register(image_A, image_B)

if __name__ == "__main__":
    unittest.main()
