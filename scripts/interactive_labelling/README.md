# Module Description: interactive_labelling

- Most of the pre-processed ground truths are not perfect.

- They are originally the analysis results of P-TRAP plus the post-correction by experts. 

- This module is used for interactive labelling. We can click on the original image to
add/remove junctions, which will later be used for annotations generation.

## Motivation

- In order to generate annotations (i.e., ground truths) for model training, we need to draw the bounding boxes around the target junctions.
- One way to do this is by getting (1) the midpoint of the bounding box and (2) the bounding box dimensions (i.e., height and width).
  1. Extracting the midpoint coordinate (x, y) from the XML file under the format *.ricepr*. This data is adapted from [AL-Tam et al. (2013)](https://link.springer.com/article/10.1186/1471-2229-13-122).
  2. Optimal bounding box size is determined through extensive experiments
- The data adapted from [AL-Tam et al. (2013)](https://link.springer.com/article/10.1186/1471-2229-13-122) has some inconsistency which may confuse the detection model. Therefore, extra data pre-preprocessing is needed.

## Objective

- **interactive_labelling** module allows for interactively clicking on non-processed ground truth images to (1) add and (2) remove objects (i.e., junctions).
- After interaction, changes are saved in a copy of the *.ricepr* file.
- Images inside `../../data/processed/` will have their names marked with `[done]` to indicate changes have been saved. To re-edit/make new changes from scratch, simply omit the `[done]` from the image name and run the previous code again. 

## Usage

```python
from scripts.interactive_labelling.InteractiveLabelling import InteractiveLabelling

# Input
img_path = "non-processed_ground_truth.jpg"
save_path = "parent_directory"

labeler = InteractiveLabelling(img_path=img_path, save_path=save_path)  # Create an instance
labeler.run()  # Show interactive figure, left mouse click to add, right mouse click to remove
labeler.show_update_img()  # Show updated image, with added and removed junctions
```

