import math
import xml.etree.ElementTree as ET


def euclidean_distance(tuple1, tuple2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(tuple1, tuple2)))


def min_distance(clicked_point, points_list):
    """
    clicked_point (tuple): (x, y) the clicked point
    points_list (list[tuple]): list of junctions
    """
    # Initialize minimum distance and corresponding tuple
    min_dist = float('inf')
    min_tuple = None
    
    # Iterate over the list of tuples to find the minimum distance and tuple
    for tup in points_list:
        dist = euclidean_distance(clicked_point, tup)
        if dist < min_dist:
            min_dist = dist
            min_tuple = tup
    
    return min_dist, min_tuple


def update_ricepr(PATH, update):
    """
    PATH: path to .ricepr file
    update (dict): {'remove': [], 'add': []}
    """
    tree = ET.parse(PATH)
    root = tree.getroot()
    
    vertices = root.find('.//vertices')
    num_vertices = len(vertices.findall('vertex'))

    # Removing junctions
    for i, code in enumerate(update["remove"]):
        print(f"Removed\tvertex {i+1}/{len(update['remove'])}")
        num_vertices -= 1
        for vertex in vertices.findall('vertex'):
            vertex_str = ET.tostring(vertex, encoding='unicode').strip()
            if vertex_str == code:
                vertices.remove(vertex)
                
    assert num_vertices == len(vertices.findall('vertex')), "(1) You clicked on generating junctions. (2) At least 2 clicked points have the same nearest neighbor. Thus, fewer vertices than specified were removed. Try again if needed."

    # Adding junctions
    for i, vertex in enumerate(update["add"]):
        print(f"Added\tvertex {i+1}/{len(update['add'])}")
        vertex_tag = vertex["tag"]
        vertex_id = vertex["id"]
        vertex_x = vertex["x"]
        vertex_y = vertex["y"]
        vertex_type = vertex["type"]
        vertex_fixed = vertex["fixed"]

        new_vertex = ET.Element(
            vertex_tag,
            id=vertex_id,
            x=vertex_x,
            y=vertex_y,
            type=vertex_type,
            fixed=vertex_fixed
        )
        
        vertices.insert(0, new_vertex)
        
    ET.indent(tree)  
    tree.write(PATH, encoding='utf-8', xml_declaration=True)

    print(f"Saved {len(update['add']) + len(update['remove'])} changes to {PATH}")
