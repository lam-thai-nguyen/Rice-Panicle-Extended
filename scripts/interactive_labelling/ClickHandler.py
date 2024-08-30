import shutil
from PIL import Image
from matplotlib import pyplot as plt
from ..generate_annotations.riceprManager import riceprManager
from .helpers import min_distance, update_ricepr


class ClickHandler:
    def __init__(self, img_path, ricepr_path):
        self.ricepr_path = ricepr_path  # .ricepr file
        self.filename = self.ricepr_path.split("/")[-1]
        self.species = self.ricepr_path.split("/")[-2]
        
        self.img_path = img_path
        self.fig, self.ax = plt.subplots(figsize=(10, 9))
        self.img = Image.open(self.img_path)
        self.ax.imshow(self.img)
        self.ax.axis("off")
        
        self.ricepr_manager = riceprManager(PATH=ricepr_path)
        self.junctions = self.ricepr_manager.read_ricepr()[0]
        self.generating = self.junctions.return_generating()
        self.primary = self.junctions.return_primary()
        self.secondary = self.junctions.return_secondary()
        self.tertiary = self.junctions.return_tertiary()
        self.quaternary = self.junctions.return_quaternary()

        self.addition = list()
        self.removal = list()
        self.update = {"add": [], "remove": []}

    def get_update(self):
        return self.update
    
    def update_ricepr(self) -> str:
        """Important function. Be cautious before calling"""
        # Step 1: Copy .ricepr file
        src = self.ricepr_path
        dst = f"data/processed/{self.species}/{self.filename}"
        print(f"==>> ClickHandler - Creating a copy of {src}")
        shutil.copy(src, dst)
        
        # Step 2: Modify .ricepr file
        print("==>> ClickHandler - Modifying vertices")
        self.find_nearest()
        update_ricepr(dst, self.get_update())
        
        return dst
        
    
    def onclick(self, event):
        """
        Left mouse click: Add junction
        Right mouse click: Remove junction nearest to the clicked point
        """
        if event.inaxes and event.button == 1:  # left mouse button
            x = int(event.xdata)
            y = int(event.ydata)
            if (x, y) not in self.addition:
                self.addition.append((x, y))
                self.ax.plot(x, y, 'bo', markersize=5)  # Add blue dot
                self.fig.canvas.draw()
                print(f"Added junction at ({x}, {y})")
            
        if event.inaxes and event.button == 3:  # Right mouse button
            x = int(event.xdata)
            y = int(event.ydata)
            if (x, y) not in self.removal:
                self.removal.append((x, y))
                self.ax.plot(x, y, 'rx', markersize=6)  # Add red x
                self.fig.canvas.draw()
                print(f"Removed junction nearest to ({x}, {y})")
            
    def find_nearest(self):
        """Find nearest neighbor and encode"""
        junctions = self.junctions.return_junctions()

        for x, y in self.addition:
            _, neighbor = min_distance(clicked_point=(x, y), points_list=junctions)
            self.encode_add(clicked_point=(x, y), neighbor=neighbor)

        for x, y in self.removal:
            _, neighbor = min_distance(clicked_point=(x, y), points_list=junctions)
            self.encode_remove(neighbor)
            
    def encode_add(self, clicked_point, neighbor) -> None:
        x, y = clicked_point
        level = None
        
        if neighbor in self.generating:
            level = "Generating"
        elif neighbor in self.primary:
            level = "Primary"
        elif neighbor in self.secondary:
            level = "Seconday"
        elif neighbor in self.tertiary:
            level = "Tertiary"
        elif neighbor in self.quaternary:
            level = "Quaternary"
            
        code = {
            "tag": 'vertex',
            "id": f"java.awt.Point[x={str(x)},y={str(y)}]",
            "x": str(x),
            "y": str(y),
            "type": level,
            "fixed": "false"
        }
        
        self.update["add"].append(code)
        
    def encode_remove(self, neighbor) -> None:
        x, y = neighbor
        level = None
        
        if neighbor in self.generating:
            level = "generating"
        elif neighbor in self.primary:
            level = "primary"
        elif neighbor in self.secondary:
            level = "secondary"
        elif neighbor in self.tertiary:
            level = "tertiary"
        elif neighbor in self.quaternary:
            level = "quaternary"
            
        level2type = {
            "generating": "Generating",
            "primary": "Primary",
            "secondary": "Seconday",
            "tertiary": "Tertiary",
            "quaternary": "Quaternary"
        }
        
        code = f'<vertex id="java.awt.Point[x={x},y={y}]" x="{x}" y="{y}" type="{level2type[level]}" fixed="false" />'
        self.update["remove"].append(code)
        