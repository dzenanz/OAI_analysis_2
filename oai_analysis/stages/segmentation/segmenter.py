#!/usr/bin/env python
"""
Created by zhenlinx on 1/19/19
"""
from abc import ABC, abstractmethod
import torch

from oai_analysis.stages.segmentation.image_transforms import Partition
from oai_analysis.stages.segmentation.module_parameters import ParameterDict
from oai_analysis.stages.segmentation.networks import get_network
from oai_analysis.stages.segmentation.utils import initialize_model

def load_json_to_dict(json_file):
    para = ParameterDict()
    para.load_JSON(json_file)
    return para.ext

class Segmenter(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.model = None
        self.config = None

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass

    @abstractmethod
    def segment(self, *args, **kwargs):
        pass


class Segmenter3DInPatch(Segmenter):

    def __init__(self, mode=None, config=None):
        super(Segmenter3DInPatch, self).__init__()
        self.config = config
        self.ready = False

    def training_setup(self, config):
        pass

    def testing_setup(self, config):
        pass

    def pred_setup(self):
        training_config = load_json_to_dict(self.config["training_config_file"])
        self.partition = Partition(training_config["patch_size"], self.config["overlap_size"],
                                                 padding_mode='reflect', mode="pred")
        # setup model
        self.model = get_network(training_config["model"])(**training_config["model_setting"])
        self.device = torch.device(self.config["device"])
        initialize_model(self.model, ckpoint_path=self.config["ckpoint_path"])
        self.model.to(self.device)

        self.model.eval()
        self.ready = True

    def train(self, model, training_data_loader, validation_data, optimizer, criterion, device, config,
              scheduler=None):
        train_segmentation(model, training_data_loader, validation_data, optimizer, criterion, device, config,
                           scheduler=None)

    def test(self, *args, **kwargs):
        pass

    def segment(self, image):
        pass


class CascadedSegmenter(Segmenter3DInPatch):
    def training_init(self):
        super(CascadedSegmenter, self).training_init()

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass

    def segment(self, *args, **kwargs):
        pass


class Segmenter3DInPatchClassWise(Segmenter3DInPatch):
    def __init__(self, mode=None, config=None):
        super(Segmenter3DInPatchClassWise, self).__init__(mode, config)
    
    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass

    def segment(self, image, if_output_prob_map=False, if_output_itk=True):
        if not self.ready:
            self.pred_setup()
        sample = {'image': image, 'name': ''}
        image_tiles = self.partition(sample)['image']
        outputs = []
        with torch.no_grad():
            # one patch at a time: it keeps a GPU busy, and 4 at a time were no faster but reserved up to 14 GB of GPU
            # memory instead of 4.6 GB, which on a 16 GB GPU, alternating with registration, made it up to 6x slower
            for patch in image_tiles.split(1):
                outputs.append(self.model(patch.to(self.device)).cpu())

            predictions = torch.sigmoid(torch.cat(outputs, dim=0))

            if not if_output_prob_map:
                predictions = predictions > 0.5

            pred_assemble_FC = self.partition.assemble(predictions[:, 0, :, :, :], if_itk=if_output_itk,
                                                       crop_size=self.config["overlap_size"])
            pred_assemble_TC = self.partition.assemble(predictions[:, 1, :, :, :], if_itk=if_output_itk,
                                                       crop_size=self.config["overlap_size"])

        return pred_assemble_FC, pred_assemble_TC
