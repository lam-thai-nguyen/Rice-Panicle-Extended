import os
from ..generate_annotations.AnnotationsGenerator import AnnotationsGenerator


def junctions2img(img_path: str, ricepr_path: str, bbox_size: int, save_path: str, skeleton_based=False, oriented_method=0):
    """
    A utils function to interact with *generate_annotations* module
    
    This function save an image with the bounding boxes that capture junctions of a rice panicle

    Args:
        img_path (str): original image path
        ricepr_path (str): .ricepr path
        save_path (str): the parent dir. (file name will be img_name_junctions.jpg)
    """
    generator = AnnotationsGenerator(img_path, ricepr_path, bbox_size)
    generator.generate_junctions(
        save_path_img=save_path,
        show=False,
        skeleton_based=skeleton_based,
        oriented_method=oriented_method,
    )
    
    
if __name__ == "__main__":
    original_img_path = "data/raw"
    african_path = original_img_path + "/African"
    asian_path = original_img_path + "/Asian"
    
    processed_path = "data/processed"
    processed_african_path = processed_path + "/African"
    processed_asian_path = processed_path + "/Asian"
    
    remove_end_generating = False  # Change this if needed
    save_path = "test"  # Change this if needed
    
    # =============================== #
    #      Comment out if needed      #
    # =============================== #
    
    for original_img in os.listdir(african_path):
        if original_img.endswith(".jpg"):
            name = original_img.split(".")[0]
            img_path = african_path + "/" + original_img
            ricepr_path = processed_african_path + f"/{name}.ricepr"
            junctions2img(
                img_path=img_path,
                ricepr_path=ricepr_path,
                bbox_size=26,  # Change this if needed
                save_path="data/annotations/African/oriented",  # Change this if needed
                skeleton_based=False,
                oriented_method=0,
            )
            break  # Comment out if needed
    
    for original_img in os.listdir(asian_path):
        if original_img.endswith(".jpg"):
            name = original_img.split(".")[0]
            img_path = asian_path + "/" + original_img
            ricepr_path = processed_asian_path + f"/{name}.ricepr"
            junctions2img(
                img_path=img_path,
                ricepr_path=ricepr_path,
                bbox_size=26,  # Change this if needed
                save_path="data/annotations/Asian/oriented",  # Change this if needed
                skeleton_based=False,
                oriented_method=0,
            )
            break  # Comment out if needed
            