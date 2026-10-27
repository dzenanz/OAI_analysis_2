import warnings

import oai_analysis.registration
from oai_analysis.atlas import Atlas
from oai_analysis.pipeline import default_device
from oai_analysis.stages.segmentation import UNetSegmenter


class AnalysisObject:
    """Deprecated: use OAPipeline, or the stages in oai_analysis.stages directly."""

    def __init__(self):
        warnings.warn("AnalysisObject is deprecated; use OAPipeline or oai_analysis.stages instead",
                      DeprecationWarning, stacklevel=2)
        self.device = default_device()
        self.segmenter = UNetSegmenter()
        self.segmenter.setup(self.device)

        # The GradICON knee registerer is only created when register() is called:
        # the pipeline registers with uniGradICON instead.
        self.registerer = None

    @property
    def atlas_image(self):
        return Atlas.default().image

    def segment(self, preprocessed_image):
        probability = self.segmenter.run(preprocessed_image)
        return (probability["FC"], probability["TC"])

    def register(self, preprocessed_image):
        if self.registerer is None:
            self.registerer = oai_analysis.registration.ICON_Registration()
        registration = self.registerer.register(preprocessed_image, self.atlas_image)
        return registration
