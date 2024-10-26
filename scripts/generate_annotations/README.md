# Module Description: generate_annotations

## Structure

- The **generate_annotations** module contains classes about subjects that are related to the annotation process.
- These classes include:
  - main class that is most frequently used
    - `AnnotationGenerator`
  - classes about rice panicles
    - `riceprManager`
    - `Junctions`
    - `Edges`
  - classes about annotation (i.e., bounding box)
    - `HorizontalBox` 
    - `OrientedBox`
    - `SkeletonBasedBox`

## Class Description

### Main class that is most frequently used

- `AnnotationGenerator` is the most straightforward ad intuitive to users. It combines the properties, functionalities of other classes.

### Classes about rice panicles

- Each rice panicle comes with the following properties:
  - **.ricepr** file that comes with the project (originally from [AL-Tam et al. (2013)](https://link.springer.com/article/10.1186/1471-2229-13-122)).
  - **junctions** are intersection points of different branches. In other words, a junction is the center of a crossroad where each road is a branch.
  - **edges** (less important than junctions in this project) are lines connecting adjacent junctions. Edges are roads.

### Classes about annotation (i.e., bounding box)

- In this project, we don't have to manually drag a bounding box for every image to get **one** version of annotation. 
- Instead, we make use of the junction position and generate **several** versions of annotation.
- Three options are:
  - Horizontal bounding box (HBB).
  - Oriented bounding box (OBB).
  - Skeleton-based bounding box (SBB).

## Usage

```python
"""
Draw bounding box on original image for visual inspection
Frequently used in development process as well as documenting and presenting.
"""

# Ensuring successful import
import sys
import os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)

# Used module
from scripts.generate_annotations.AnnotationsGenerator import AnnotationsGenerator

# Create instance
generator = AnnotationsGenerator(
    img_path="data/raw/Asian/10_2_1_1_1_DSC01291.jpg",
    ricepr_path="data/processed/Asian/10_2_1_1_1_DSC01291.ricepr",
    bbox_size=26,
)

# Draw
generator.draw_junctions(
  save_path=".",
  show=False,
  skeleton_based=False,  # True for skeleton based box
  oriented_method=0,  # choice={0: HBB, 1: OBBv1, 2: OBBv2}
)

# NOTE: ../utils/junctions2img.py performs this operation.
```

---

```python
"""
Encode bounding box and export to .txt file for model training.
"""
# ongoing
```
