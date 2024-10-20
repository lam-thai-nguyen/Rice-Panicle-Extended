"""
The math here is confusing because of the discrepancy between Cartesian system and OpenCV
- Cartesian: upward y axis
- OpenCV image: downward y axis
"""

import math


class OrientedBox:
    def __init__(self, junctions: list) -> None:
        self.junctions = junctions
        self.neighbor = list()
        self.theta = list()  
        self.rect = list()
        
    def run(self, width, height, method: int) -> list:
        self.find_neighbor()
        self.find_theta(method)
        self.find_rect(width, height)
        return self.rect
    
    def find_neighbor(self):
        for junction in self.junctions:
            _, neighbor = self._nearest_neighbor(junction)
            self.neighbor.append(neighbor)
    
    def find_theta(self, method: int):
        """
        method = {1, 2}
            - 1: box vertex lies on the line connecting 2 junctions.
            - 2: box midline and the line connecting 2 junctions are coincident.
        """
        for pt1, pt2 in zip(self.junctions, self.neighbor):
            x1, y1 = pt1
            x2, y2 = pt2
            
            # angle_rad in (-pi/2, pi/2], angle_deg in (-180, 180]
            # This angle is counterclockwise in Cartesian system, clockwise in OpenCV.
            # Visually, we compute the clockwise rotating angle here.
            angle_rad = math.atan2(y2-y1, x2-x1)
            angle_deg = math.degrees(angle_rad)
            
            # Disregard vector orientation to make sure angle_deg falls in [0., 180.]
            if angle_deg < 0:
                angle_deg += 180.
            
            # Theta is the counterclockwise angle in Cartesian system, clockwise in OpenCV.
            # As the box vertex lies on the connecting line, we subtract 45. from theta to get the rotating angle
            if method == 1:
                angle_diagonal = 45.
                theta = angle_deg - angle_diagonal
            elif method == 2:
                theta = angle_deg
                
            self.theta.append(theta)
            
    def find_rect(self, width, height):
        for center, angle in zip(self.junctions, self.theta):
            self.rect.append(
                tuple(
                    [center, (width, height), angle]
                )
            )
            
    def _euclidean_distance(self, t1, t2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(t1, t2)))

    def _nearest_neighbor(self, junction, junction_list=None):
        if junction_list is None:
            junction_list = self.junctions
            
        min_dist = float('inf')
        nearest_neighbor = None
        
        for neighbor in junction_list:
            if neighbor == junction:
                continue
            
            dist = self._euclidean_distance(junction, neighbor)
            if dist < min_dist:
                min_dist = dist
                nearest_neighbor = neighbor
        
        return min_dist, nearest_neighbor
            