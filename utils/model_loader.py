import pickle
import os

def load_model(model_path):
    """
    Loads a serialized model from the given path.
    Args:
        model_path (str): The path to the .pkl file.
    Returns:
        The deserialized object.
    """
    try:
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
