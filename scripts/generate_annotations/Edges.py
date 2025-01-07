class Edges:
    """an edge manager for an image"""
    num_entry = 0
    
    def __init__(self):
        self.entries = []
    
    def __len__(self):
        return self.num_entry
    
    def __getitem__(self, index):
        return self.entries[index]
    
    def add(self, edges) -> None:
        """
        Args:
            edges (list): [(x1, y1, x2, y2), ...]
        """
        for edge in edges:
            assert len(edge) == 4, "Incorrect edge format"
            edge = tuple(edge)
            self.entries.append(edge)
            self.num_entry += 1
            
        
def test():
    edges = Edges()
    edges.add([(1, 2, 3, 4), (5, 6, 7, 8)])
    assert edges.num_entry == 2
    assert len(edges) == 2
    assert len(edges[0]) == 4
    print("All tests passed")
    

if __name__ == "__main__":
    test()
    