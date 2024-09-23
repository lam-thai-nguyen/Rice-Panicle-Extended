import os


def compute_num_objects(split_path):
    """Compute the number of objects in a dataset"""
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
    split_path = "data/splits/split15"
    num_obj, obj_aver = compute_num_objects(split_path)
    print(f"==>> num_obj: {num_obj}")
    print(f"==>> obj_aver: {obj_aver:.2f}")
    