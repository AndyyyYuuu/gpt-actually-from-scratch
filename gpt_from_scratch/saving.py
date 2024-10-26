import pickle 
from .utils import _enforce_type

def save(value: dict, location: str) -> None: 
    _enforce_type(value, dict)
    _enforce_type(location, str)
    with open(location, 'wb') as file: 
        pickle.dump(value, file)
    file.close()

def load(location: str) -> dict: 
    _enforce_type(location, str)
    with open(location, 'rb') as file: 
        value = pickle.load(file)
    file.close()
    return value
