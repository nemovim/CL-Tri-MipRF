from .RFModel import RFModel
from .trimipRF import TriMipRFModel
from .CLTrimipRF import CLTriMipRFModel


def get_model(model_name: str = 'Tri-MipRF') -> RFModel:
    if 'Tri-MipRF' == model_name:
        return TriMipRFModel
    elif 'CL-Tri-MipRF' == model_name:
        return CLTriMipRFModel
    else:
        raise NotImplementedError
