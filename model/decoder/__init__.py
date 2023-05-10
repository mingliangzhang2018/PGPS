# from .transformer import TransformerModel
from config import decoder_list
from .rnn_decoder import DecoderRNN
from .tree_decoder import TreeDecoder
from .transformer import TransformerDecoder

def get_decoder(params, *args):
         
    if not params.decoder_type in decoder_list:
        raise NotImplementedError(
            "Unsupported Classifier: {}".format(params.decoder_type))

    if params.decoder_type == "transformer":
        decoder = TransformerDecoder(params, *args)
    elif params.decoder_type == "rnn_decoder":
        decoder = DecoderRNN(params, *args)
    elif params.decoder_type == "tree_decoder":
        decoder = TreeDecoder(params, *args)
    else:
        raise NotImplementedError("Unsupported Decoder: {}".format(params.decoder_type))
             
    return decoder


