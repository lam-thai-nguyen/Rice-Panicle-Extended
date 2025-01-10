import math
import xml.etree.ElementTree as ET
from .Junctions import Junctions
from .Edges import Edges


class riceprManager:
    def __init__(self, PATH):
        """
        Create a manager for .ricepr files.

        Args:
            PATH (str): .ricepr file path
        """
        self.PATH = PATH
        self.name = self.PATH.split("/")[-1].split(".")[0]
        self.species = "Asian" if "Asian" in PATH else "African" if "African" in PATH else None
        self.format = ".ricepr"
        self.junctions = Junctions()
        self.edges = Edges()
    
    def read_ricepr(self) -> tuple:
        """
        Read a given .ricepr file and return its content
        
        Returns:
            tuple: (Junctions, Edges)
        """
        print(f"==>> riceprManager - Reading {self.name + self.format}")
        junctions = self._get_junctions()
        for level in junctions:
            for coord in junctions[level]:
                self.junctions.add(level=level, coord=coord)
            
        edges = self._get_edges()
        self.edges.add(edges=edges)
        
        return (self.junctions, self.edges)
        
    def _get_junctions(self) -> dict:
        """
        Returns the junctions in the given .ricepr file
        """
        tree = ET.parse(self.PATH)
        root = tree.getroot()
        
        junctions = {
            "generating" : [],
            "terminal" : [],
            "primary" : [],
            "secondary" : [],
            "tertiary" : [],
            "quaternary" : []
        }
        
        for vertex in root.iter('vertex'):
            x = int(vertex.attrib['x'])
            y = int(vertex.attrib['y'])
            type_ = vertex.attrib['type']
            coord = tuple([x, y])
            if type_ == "Generating":
                junctions["generating"].append(coord)
            elif type_ == "End":
                junctions["terminal"].append(coord)
            elif type_ == "Primary":
                junctions["primary"].append(coord)
            elif type_ == "Seconday":
                junctions["secondary"].append(coord)
            elif type_ == "Tertiary":
                junctions["tertiary"].append(coord)
                
        return junctions
    
    def _get_edges(self) -> list:
        """
        Returns the edges in the given .ricepr file. Edges are the lines connecting two junctions.
        """
        tree = ET.parse(self.PATH)
        root = tree.getroot()
        
        edges = []
        for edge in root.iter('edge'):
            vertex1 = edge.attrib['vertex1']
            vertex2 = edge.attrib['vertex2']
            
            x1 = int(vertex1.split('=')[1].split(',')[0])
            y1 = int(vertex1.split('=')[2].split(']')[0])
            
            x2 = int(vertex2.split('=')[1].split(',')[0])
            y2 = int(vertex2.split('=')[2].split(']')[0])
            
            edge = tuple([x1, y1, x2, y2])
            edges.append(edge)

        return edges

    # These 3 functions below are placed here instead of inside Edges.py because
    # the info. from edges only is not enough, but we also need to incorporate info. from junctions
    # FYI [very important]: end points (terminals) only appear as the second point (vertex2) in .ricepr file
    def get_grains(self) -> list:
        """Returns grains in the format of (x1, y1, x2, y2)"""
        assert len(self.edges) > 0, "You need at least one edge to do this operation"

        grains = list()
        for edge in self.edges:
            _, _, x2, y2 = edge
            if (x2, y2) in self.junctions.return_terminal():
                grains.append(edge)
        
        return grains
        
    def get_primary_branches(self) -> list:
        """Returns primary branches in the format of (x1, y1, x2, y2)"""
        
        assert len(self.edges) > 0, "You need at least one edge to do this operation"

        # Some ideas to identify a primary branch
        # - Starts with a primary junction or generating and ends with a terminal (end point)
        # - If we connect the edges from the 2 points (start and end points), we only see the edges pass through primary, secondary and terminal (no tertiary and beyond)
        # - 
        
        primary_branches = list()
        
        def find_parent(vertex, parents=None) -> list:
            """Use recursion to find the parent, vertex is (x, y)"""
            if parents is None:
                parents = [vertex]
            
            for edge in self.edges:
                # If vertex is the second point of this edge
                if vertex == edge[2:]:
                    parent = edge[0:2]
                    break

            parents.append(parent)
            vertex = parent

            if vertex not in self.junctions.return_primary() and vertex not in self.junctions.return_generating():
                return find_parent(vertex, parents)
            else:
                return parents

        # Inspect the family tree from end point (upwards) to primary junction -> parents = [endpoint, ..., primary junction]
        family_tree = list()
        for endpoint in self.junctions.return_terminal():
            parents = find_parent(endpoint)
            family_tree.append(parents)
        
        # There are some outlier cases where they are not primary branches. That is when an end point only has one parent that is a primary junction.
        # Clean family tree
        family_tree = [i for i in family_tree if len(i) > 2]
        
        # Identify which set of end points belongs to the same branch, and store them in `clusters`
        root_nodes = list(set([tree[-1] for tree in family_tree]))
        clusters = list()
        for root_node in root_nodes:
            candidates = [family_tree[i] for i in range(len(family_tree)) if family_tree[i][-1] == root_node]

            # Classify and store inside `clusters` -> the problem is: given a nested list, find inner lists that have 2 mutual entries.
            for i, inner1 in enumerate(candidates):  # inner1 is a list
                # Initialize a new empty list. A cluster is composed of different paths that all end up at the same primary junction.
                cluster = list()
                
                # Check if visited
                if len(clusters) > 0:
                    if inner1 in clusters[-1]:
                        continue
                cluster.append(inner1)
                
                for inner2 in candidates[i + 1:]:  # inner2 is a list
                    if len(clusters) > 0:
                        if inner2 in clusters[-1]:
                            continue
                    if len(set(inner1) & set(inner2)) >= 2:
                        cluster.append(inner2)

                clusters.append(cluster)

        # Take the furthest end point from the primary junction to be the tip of that branch
        for cluster in clusters:
            primary_junction = cluster[0][-1]
            endpoints = [path[0] for path in cluster]

            max_dist = - math.inf
            furthest_endpoint = None
            for endpoint in endpoints:
                dist = math.dist(endpoint, primary_junction)
                if dist > max_dist:
                    max_dist = dist
                    furthest_endpoint = endpoint

            primary_branches.append([furthest_endpoint, primary_junction])
                    
        return primary_branches
        
    # NOTE: The logic here might not be correct. Needs confirmation from Stefan. Not every branch that starts with secondary junction and ends with terminal is a secondary branches
    def get_secondary_branches(self) -> list:
        """Returns secondary branches in the format of (x1, y1, x2, y2)"""
        assert len(self.edges) > 0, "You need at least one edge to do this operation"

        secondary_branches = list()
        for edge in self.edges:
            x1, y1, x2, y2 = edge
            if (x1, y1) in self.junctions.return_secondary() and (x2, y2) in self.junctions.return_terminal():
                secondary_branches.append(edge)
                
        return secondary_branches
        