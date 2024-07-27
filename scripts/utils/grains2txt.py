import os
from ..generate_annotations.AnnotationsGenerator import AnnotationsGenerator


def grains2txt(img_path: str, ricepr_path: str, save_path: str):
    """
    A utils function to interact with *generate_annotations* module
    
    This function save a text file with the encoded bounding boxes that capture grains of a rice panicle

    Args:
        img_path (str): original image path
        ricepr_path (str): .ricepr path
        save_path (str): the parent dir. (file name will be img_name_grains.txt)
    """
    generator = AnnotationsGenerator(img_path=img_path, ricepr_path=ricepr_path)
    generator.encode_grains(save_path=save_path)
    
    
if __name__ == "__main__":
    original_img_path = "data/raw"
    african_path = original_img_path + "/African"
    asian_path = original_img_path + "/Asian"
    
    # =============================== #
    #      Comment out if needed      #
    # =============================== #
    
    for original_img in os.listdir(african_path):
        if original_img.endswith(".jpg"):
            name = original_img.split(".")[0]
            img_path = african_path + "/" + original_img
            ricepr_path = african_path + f"/{name}.ricepr"
            grains2txt(
                img_path=img_path,
                ricepr_path=ricepr_path,
                save_path="data/annotations/African",
            )
    
    
    for original_img in os.listdir(asian_path):
        if original_img.endswith(".jpg"):
            name = original_img.split(".")[0]
            img_path = asian_path + "/" + original_img
            ricepr_path = asian_path + f"/{name}.ricepr"
            grains2txt(
                img_path=img_path,
                ricepr_path=ricepr_path,
                save_path="data/annotations/Asian",
            )
            