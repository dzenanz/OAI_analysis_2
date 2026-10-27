import numpy as np
import icon_registration.config
import icon_registration.itk_wrapper
import torch
from icon_registration.pretrained_models.lung_ct import init_network

from oai_analysis import data

GRADICON_KNEE_INPUT_SHAPE = [1, 1, 80, 192, 192]


def gradicon_knee_model():
    """
    The GradICON model trained on OAI knees, equivalent to icon_registration's OAI_knees_gradICON_model(),
    but with the weights from the oai_analysis data cache instead of ./network_weights.
    """
    net = init_network("knee", pretrained=False)
    trained_weights = torch.load(data.gradicon_knee_weights(), map_location=torch.device("cpu"))
    net.regis_net.load_state_dict(trained_weights, strict=False)
    net.assign_identity_map(GRADICON_KNEE_INPUT_SHAPE)
    net.to(icon_registration.config.device)
    net.eval()
    return net


class ICON_Registration:
    def __init__(self):
        self.register_module = gradicon_knee_model()

    def register(self, fixed_image, moving_image):
        print("fixed range", np.min(fixed_image), np.max(fixed_image))
        print("moving range", np.min(moving_image), np.max(moving_image))
        phi_fixed_moving, phi_moving_fixed = icon_registration.itk_wrapper.register_pair(self.register_module, fixed_image, moving_image)

        return phi_fixed_moving
