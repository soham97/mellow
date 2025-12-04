def get_model_class(model_type):
    if model_type == 'Mellow':
        from models.mellow import Mellow
        return Mellow
    else:
        raise NotImplementedError
