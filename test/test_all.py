import unittest

import itk
import oai_analysis.registration
from oai_analysis import qc
from oai_analysis.data import test_data_dir as data_dir
from oai_analysis.algorithms import mesh_processing as mp
from oai_analysis.atlas import Atlas
from oai_analysis.pipeline import default_device
from oai_analysis.stages import UNetSegmenter
import numpy as np

class TestOAIAnalysis(unittest.TestCase):

    def setUp(self):
        self.data = data_dir()

    def test_SegmentationCPU(self):
        input_image = itk.imread(self.data / "colab_case" / "image_preprocessed.nii.gz")

        correct_FC_segmentation = itk.imread(self.data / "colab_case" / "FC_probmap.nii.gz")
        correct_TC_segmentation = itk.imread(self.data / "colab_case" / "TC_probmap.nii.gz")

        segmenter = UNetSegmenter()
        segmenter.setup(default_device())
        probability = segmenter.run(input_image)
        FC, TC = probability["FC"], probability["TC"]

        # The reference maps were made on a CPU, which reproduces them to 1e-5; a GPU differs by up to 0.0025
        for result, correct in ((FC, correct_FC_segmentation), (TC, correct_TC_segmentation)):
            result, correct = itk.array_from_image(result), itk.array_from_image(correct)
            self.assertLess(np.abs(result - correct).max(), 0.005)
            inside, correct_inside = result > 0.5, correct > 0.5
            dice = 2 * (inside & correct_inside).sum() / (inside.sum() + correct_inside.sum())
            self.assertGreater(dice, 0.999)

    def test_MeshThicknessCPU(self):
        input_image = itk.imread(self.data / "colab_case" / "image_preprocessed.nii.gz", itk.D)
        atlas_image = Atlas.default().image

        correct_FC_segmentation = itk.imread(self.data / "colab_case" / "FC_probmap.nii.gz", itk.D)
        correct_TC_segmentation = itk.imread(self.data / "colab_case" / "TC_probmap.nii.gz", itk.D)

        inner_mesh_fc_atlas = Atlas.default().inner_mesh["FC"]
        inner_mesh_tc_atlas = Atlas.default().inner_mesh["TC"]

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

        phi_AB = oai_analysis.registration.ICON_Registration().register(input_image, atlas_image)

        warped_image_FC = deform_probmap(phi_AB, input_image, atlas_image, correct_FC_segmentation)
        warped_image_TC = deform_probmap(phi_AB, input_image, atlas_image, correct_TC_segmentation)

        fc_mesh_itk = mp.get_mesh_from_probability_map(warped_image_FC)
        fc_mesh = mp.itk_mesh_to_vtk_mesh(fc_mesh_itk)
        fc_inner_atlas, fc_outer_atlas = mp.get_split_mesh(fc_mesh, inner_mesh_fc_atlas, mesh_type='FC')
        fc_inner, fc_outer = mp.get_distance(fc_inner_atlas, fc_outer_atlas)
        distance_inner_FC = mp.get_itk_mesh(fc_inner)

        tc_mesh_itk = mp.get_mesh_from_probability_map(warped_image_TC)
        tc_mesh = mp.itk_mesh_to_vtk_mesh(tc_mesh_itk)
        tc_inner_atlas, tc_outer_atlas = mp.get_split_mesh(tc_mesh, inner_mesh_tc_atlas, mesh_type='TC')
        tc_inner, tc_outer = mp.get_distance(tc_inner_atlas, tc_outer_atlas)
        distance_inner_TC = mp.get_itk_mesh(tc_inner)

        # 55366 and 19329 points on a GPU, 55406 and 19323 on a CPU
        self.assertTrue(55200 <= distance_inner_FC.GetNumberOfPoints() <= 55600)
        self.assertTrue(19250 <= distance_inner_TC.GetNumberOfPoints() <= 19400)

    def test_RegistrationCPU(self):
        input_image = itk.imread(self.data / "colab_case" / "image_preprocessed.nii.gz")

        atlas_image = Atlas.default().image

        registration = oai_analysis.registration.ICON_Registration().register(input_image, atlas_image)

        # avsm/inv_transform_to_atlas.nii.gz is from a different registration method, so it is no reference;
        # check instead that the transform does not fold space and brings the image onto the atlas
        self.assertLess(qc.folding_fraction(registration), 0.01)
        self.assertGreater(qc.similarity_to_atlas(input_image, atlas_image, registration), 0.75)

class TestImports(unittest.TestCase):

    def test_ImportsCPU(self):
        from oai_analysis import OAPipeline, CaseInfo

if __name__ == "__main__":
    unittest.main()
