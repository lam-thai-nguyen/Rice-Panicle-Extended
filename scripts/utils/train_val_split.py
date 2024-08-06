import os
import random
import shutil


def train_val_split(mode: str, root_dir: str, save_dir: str, val_size: float, random_state: int = 42, shuffle: bool = True):
    """
    Split the dataset from data/annotations to train and dev sets inside data/splits
    
    Method: Combine + Shuffle + Split + Copy

    Args:
        mode (str): "junctions" or "grains" or "all"
        root_dir (str): Parent dir that contains African/ and Asian/
        save_dir (str): Parent dir that contains train/ and val/
        val_size (float): 0. to 1.
        random_state (int): Defaults to 42.
        shuffle (bool): Defaults to True.
    
    Returns:
        train (list): List of filenames for the training set.
        val (list): List of filenames for the validation set.
    """
    assert mode in ["junctions", "grains", "all"], "Invalid mode"
    assert 0. <= val_size <= 1., "Invalid test_size"
    
    african_path = root_dir + "/African"
    asian_path = root_dir + "/Asian"
    rice_panicles = []
    
    # Combine
    for file_name in os.listdir(african_path):
        if file_name.endswith("_junctions.jpg"):
            name = file_name[:-len("_junctions.jpg")]
            rice_panicles.append(name)

    for file_name in os.listdir(asian_path):
        if file_name.endswith("_junctions.jpg"):
            name = file_name[:-len("_junctions.jpg")]
            rice_panicles.append(name)
           
    # Shuffle 
    random.seed(random_state)
    if shuffle:
        random.shuffle(rice_panicles)
    
    # Split
    split_index = int(len(rice_panicles) * val_size)
    val = rice_panicles[:split_index]
    train = rice_panicles[split_index:]
    
    # Move
    if mode == "junctions":
        _process_files(mode="junctions", val=val, train=train, save_dir=save_dir)
    elif mode == "grains":
        _process_files(mode="grains", val=val, train=train, save_dir=save_dir)
    elif mode == "all":
        # Ongoing
        ...
        
    return train, val

    
def _process_files(mode: str, val: list, train: list, save_dir: str):
    """Copy (and rename) src file from root_dir to save_dir"""
    assert mode in ["junctions", "grains", "all"], "Invalid mode"
    postfix = f"_{mode}"
    
    african_orig_path = "data/raw/African"  # Change to "data_high_res/raw/African" for high resolution dataset
    asian_orig_path = "data/raw/Asian"  # Change to "data_high_res/raw/Asian" for high resolution dataset
    african_annotations_path = "data/annotations/African"  # Change to "data_high_res/annotations/African" for high resolution dataset
    asian_annotations_path = "data/annotations/Asian"  # Change to "data_high_res/annotations/Asian" for high resolution dataset
    
    for file_name in val:
        if os.path.exists(african_orig_path + "/" + file_name + ".jpg"):  # If this is an African rice panicle
            image_src = african_orig_path + "/" + file_name + ".jpg"
            label_src = african_annotations_path + "/" + file_name + postfix + ".txt"
            image_dst = save_dir + "/" + "val/images/" + file_name + ".jpg"
            label_dst = save_dir + "/" + "val/labels/" + file_name + ".txt" 
        else:
            image_src = asian_orig_path + "/" + file_name + ".jpg"
            label_src = asian_annotations_path + "/" + file_name + postfix + ".txt"
            image_dst = save_dir + "/" + "val/images/" + file_name + ".jpg"
            label_dst = save_dir + "/" + "val/labels/" + file_name + ".txt" 
        shutil.copy(image_src, image_dst)
        shutil.copy(label_src, label_dst)
            
    for file_name in train:
        if os.path.exists(african_orig_path + "/" + file_name + ".jpg"):  # If this is an African rice panicle
            image_src = african_orig_path + "/" + file_name + ".jpg"
            label_src = african_annotations_path + "/" + file_name + postfix + ".txt"
            image_dst = save_dir + "/" + "train/images/" + file_name + ".jpg"
            label_dst = save_dir + "/" + "train/labels/" + file_name + ".txt" 
        else:
            image_src = asian_orig_path + "/" + file_name + ".jpg"
            label_src = asian_annotations_path + "/" + file_name + postfix + ".txt"
            image_dst = save_dir + "/" + "train/images/" + file_name + ".jpg"
            label_dst = save_dir + "/" + "train/labels/" + file_name + ".txt" 
        shutil.copy(image_src, image_dst)
        shutil.copy(label_src, label_dst)


if __name__ == "__main__":
    mode = "junctions"
    
    train, val = train_val_split(
        mode=mode,
        root_dir="data/annotations",  # Change to "data_high_res/annotations" for high resolution dataset
        save_dir="data/splits",  # Change to "data_high_res/splits" for high resolution dataset
        val_size=0.3
    )
    
    print("==>> Train size:", len(train))
    print("==>> Val size:", len(val))
