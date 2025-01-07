class Junctions:
    """a junction manager for an image"""
    num_entry = 0
    
    def __init__(self):
        self.level = ["generating", "terminal", "primary", "secondary", "tertiary", "quaternary"]
        self.entries = {level : [] for level in self.level}
        
    def __len__(self):
        return self.num_entry
    
    def return_entries(self) -> dict:
        return self.entries
    
    def return_generating(self) -> list:
        return self.entries["generating"]
    
    def return_terminal(self) -> list:
        return self.entries["terminal"]
    
    def return_primary(self) -> list:
        return self.entries["primary"]
        
    def return_secondary(self) -> list:
        return self.entries["secondary"]
        
    def return_tertiary(self) -> list:
        return self.entries["tertiary"]
        
    def return_quaternary(self) -> list:
        return self.entries["quaternary"]
    
    def return_junctions(self) -> list:
        generating = self.return_generating()
        primary = self.return_primary()
        secondary = self.return_secondary()
        tertiary = self.return_tertiary()
        quaternary = self.return_quaternary()
        return generating + primary + secondary + tertiary + quaternary
    
    def add(self, level, coord) -> None:
        """
        Args:
            level (str): ["generating", "terminal", "primary", "secondary", "tertiary", "quaternary"]
            coord (tuple, list): coordinate of the junction
        """
        assert len(coord) == 2, "Invalid coord"
        assert level.lower() in self.level, "Invalid level"
        
        coord = tuple(coord)
        level = level.lower()
        
        self.num_entry += 1
        self.entries[level].append(coord)
        
    def remove_end_generating(self) -> None:
        """
        Remove the end generating junction when it is not the junction
        """
        generating = self.entries["generating"]
        assert len(generating) == 2, "Incorrect number of generating junctions"
        
        # ============================================================ #
        #   End gen. junction has smaller x than Start gen. junction   #
        # ============================================================ #
        generating = sorted(generating, key=lambda x: x[0])
        end_generating = generating[0]
        self.entries["generating"].remove(end_generating)
        assert len(self.entries["generating"]) == 1, "Incorrect number of generating junctions"
        self.num_entry -= 1
        
        
def test():
    junctions = Junctions()
    junctions.add(coord=(123, 345), level="Primary")
    assert junctions.num_entry == 1
    assert len(junctions.return_primary()) == 1
    junctions.add(coord=(110, 500), level="generating")
    junctions.add(coord=(500, 500), level="generating")
    assert junctions.num_entry == 3
    assert len(junctions.return_generating()) == 2
    junctions.remove_end_generating()
    assert junctions.num_entry == 2
    assert len(junctions.return_generating()) == 1
    assert len(junctions.entries["generating"]) == 1
    print("All tests passed")


if __name__ == "__main__":
    test()
    