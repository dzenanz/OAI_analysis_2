import torch

import oai_analysis.registration
import oai_analysis.segmentation.segmenter
from oai_analysis.atlas import Atlas
from oai_analysis.data import models_dir


class AnalysisObject:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            print("WARNING: CUDA NOT AVAILABLE, FALLING BACK TO CPU")
            self.device = "cpu"

        ## Initialize segmenter
        segmenter_config = dict(
            ckpoint_path=str(models_dir() / "segmentation_model.pth.tar"),
            training_config_file=str(models_dir() / "segmentation_train_config.pth.tar"),
            device=self.device,
            batch_size=4,
            overlap_size=(16, 16, 8),
            output_prob=True,
            output_itk=True,
        )
        self.segmenter = oai_analysis.segmentation.segmenter.Segmenter3DInPatchClassWise(
            mode="pred", config=segmenter_config
        )

        # The GradICON knee registerer is only created when register() is called:
        # the pipeline registers with uniGradICON instead.
        self.registerer = None

    @property
    def atlas_image(self):
        return Atlas.default().image

    def segment(self, preprocessed_image):
        FC_probmap, TC_probmap = self.segmenter.segment(preprocessed_image, if_output_prob_map=True, if_output_itk=True)
        return (FC_probmap, TC_probmap)

    def register(self, preprocessed_image):
        if self.registerer is None:
            self.registerer = oai_analysis.registration.ICON_Registration()
        registration = self.registerer.register(preprocessed_image, self.atlas_image)
        return registration
