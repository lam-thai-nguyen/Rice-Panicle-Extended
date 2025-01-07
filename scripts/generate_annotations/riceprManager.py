import time
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
