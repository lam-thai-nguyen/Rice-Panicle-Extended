import os
import xml.etree.ElementTree as ET


def compute_num_objects(root_dir="data/processed"):
    """Compute the number of objects in a dataset"""
    african = root_dir + "/African"
    asian = root_dir + "/Asian"
    african_junction_counter = 0
    asian_junction_counter = 0
    african_image_counter = 0
    asian_image_counter = 0
    
    
    for filename in os.listdir(african):
        if filename.endswith(".ricepr"):
            file_path = african + "/" + filename
            tree = ET.parse(file_path)
            root = tree.getroot()
            junction_count = sum(1 for vertex in root.findall('.//vertex') if vertex.get('type') != 'End')
            african_junction_counter += junction_count
            african_image_counter += 1
            
    for filename in os.listdir(asian):
        if filename.endswith(".ricepr"):
            file_path = asian + "/" + filename
            tree = ET.parse(file_path)
            root = tree.getroot()
            junction_count = sum(1 for vertex in root.findall('.//vertex') if vertex.get('type') != 'End')
            asian_junction_counter += junction_count
            asian_image_counter += 1
    
    print(f"==>> african_image_counter: {african_image_counter}")
    print(f"==>> asian_image_counter: {asian_image_counter}")

    num_obj = african_junction_counter + asian_junction_counter
    african_obj_per_image = african_junction_counter / african_image_counter
    asian_obj_per_image = asian_junction_counter / asian_image_counter
    obj_per_image = num_obj / (african_image_counter + asian_image_counter)
    
    return num_obj, african_junction_counter, asian_junction_counter, obj_per_image, african_obj_per_image, asian_obj_per_image


def compute_num_objects_training_set(split_path):
    """Compute the number of objects in a training dataset (70% of original dataset)"""
    train_path = split_path + "/train/labels"
    val_path = split_path + "/val/labels"
    counter = 0
    
    for filename in os.listdir(train_path):
        file_path = train_path + "/" + filename
        with open(file_path) as f:
            counter += len(f.readlines())
        
    for filename in os.listdir(val_path):
        file_path = val_path + "/" + filename
        with open(file_path) as f:
            counter += len(f.readlines())
    
    num_obj = counter
    obj_aver = counter / (len(os.listdir(train_path)) + len(os.listdir(val_path)))
    
    return num_obj, obj_aver


if __name__ == "__main__":
    split_path = "data/splits/split2"
    num_obj, obj_aver = compute_num_objects_training_set(split_path)
    print(f"==>> num_obj: {num_obj}")
    print(f"==>> obj_aver: {obj_aver:.2f}")

    print("".center(50, "="))

    num_obj, african_junction_counter, asian_junction_counter, obj_per_image, african_obj_per_image, asian_obj_per_image = compute_num_objects(root_dir="data/processed")
    print(f"==>> num_obj: {num_obj}")
    print(f"==>> african_junction_counter: {african_junction_counter}")
    print(f"==>> asian_junction_counter: {asian_junction_counter}")
    print(f"==>> obj_per_image: {obj_per_image}")
    print(f"==>> african_obj_per_image: {african_obj_per_image}")
    print(f"==>> asian_obj_per_image: {asian_obj_per_image}")
