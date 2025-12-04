from models.htsat import HTSATWrapper
from models.cnn14 import CNN14Wrapper

def get_audio_encoder(name: str):
    if name == "HTSAT":
        return HTSATWrapper, 768
    elif name == "Cnn14":
        return CNN14Wrapper, 2048
    else:
        raise Exception('The audio encoder name {} is incorrect or not supported'.format(name))