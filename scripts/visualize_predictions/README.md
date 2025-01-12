# Module Description: visualize_predictions

## Structure

```
visualize_predictions
├── Visualizer.py           # main class
├── plot_loss.py            # util file
└── predict_show.py         # util file
```

## Class Description

- **visualize_predictions** module helps visualize the **outputs** of [ultralytics](https://github.com/ultralytics/ultralytics) YOLO training.
- The `Visualizer` class allows us to 
  - `plot_loss(split_name)`: Convert the .csv file (training output) to a plot of graphs.
  - `predict_show(img_path, checkpoint, conf)`: Show the YOLO prediction with one extra info., that is the number of predictions and the type of BBs.
- FYI: `split_name` is the used split dataset used to train the YOLO model.

## Usage

- Refer to <tt>src/visualize_result.py</tt> for usage. This is a reusable file for this purpose.
- Below is the description of this file.

```python
"""
src/visualize_result.py -> visualize()
"""

def visualize(plot_loss: False, predict_show=False, **kwargs) -> None:
    split_name = kwargs.get("split_name", None)
    img_path = kwargs.get("img_path", None)
    checkpoint = kwargs.get("checkpoint", None)
    conf = kwargs.get("conf", None)
    
    visualizer = Visualizer()

    if plot_loss:
        visualizer.plot_loss(split_name)

    if predict_show:
        visualizer.predict_show(img_path, checkpoint, conf)
```

---

- This is how each function works.

```python
# Plot losses from results.csv file generated by ultralytics after training
visualize(plot_loss=True, split_name="split1")

# Show prediction
visualize(
    predict_show=True, 
    img_path=".../Asian/<name>.jpg", checkpoint="checkpoints/split1/best.pt", 
    conf=0.3,
)
```

---

- However, for regular use, just change the specified hyperparameters in <tt>src/visualize_result.py</tt>. 

```python
# Change this if needed

# Hyperparameters (change if needed)
plot_loss = False
predict_show = True

# Keyword Arguments (Change if static)
split_name = "split2"  # static

img_name = benchmark_image["lots_of_junctions"]  # or species/name.jpg
img_path = f"data/raw/{img_name}"

checkpoint = f"checkpoints/{split_name}/best.pt"

mode = "train"  # static
conf = confidence[split_name][0] if mode == "train" else confidence[split_name][1]
```