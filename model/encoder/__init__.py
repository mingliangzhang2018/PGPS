from .lstm import LSTM
from .gru import GRU
from config import encoder_list
from .transformer import TransformerEncoder

def get_encoder(params, *args):

    if not params.encoder_type in encoder_list:
        raise NotImplementedError(
            "Unsupported Classifier: {}".format(params.encoder_type))

    if params.encoder_type == "transformer":
        pass
    elif params.encoder_type == "lstm":
        encoder = LSTM(params, *args)
    elif params.encoder_type == "gru":
        encoder = GRU(params, *args)
    else:
        raise NotImplementedError("Unsupported Encoder: {}".format(params.encoder_type))

    return encoder
