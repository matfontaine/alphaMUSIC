from pathlib import Path
import pickle

def walk(path): 
    for p in Path(path).iterdir(): 
        if p.is_dir(): 
            yield from walk(p)
            continue
        yield p.resolve()

def save_to_pkl(path, obj):
    with open(path, 'wb') as file:
        # A new file will be created
        pickle.dump(obj, file)

def load_from_pkl(path):
    with open(path, 'rb') as file:
        # Call load method to deserialze
        return pickle.load(file)