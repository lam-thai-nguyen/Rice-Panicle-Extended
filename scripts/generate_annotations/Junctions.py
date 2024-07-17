class Junctions:
    num_entry = 0
    num_generating = 0
    num_terminal = 0
    num_primary = 0
    num_secondary = 0
    num_tertiary = 0
    num_quaternary = 0
    
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
    
    def add(self, level, coord) -> None:
        """
        Args:
            level (str): ["generating", "terminal", "primary", "secondary", "tertiary", "quaternary"]
            coord (tuple, list): coordinate of the junction
        """
        assert type(coord) in [list, tuple], "Invalid type"
        assert len(coord) == 2, "Invalid coord"
        assert type(level) == str, "Invalid type"
        assert level.lower() in self.level, "Invalid level"
        
        coord = tuple(coord)
        level = level.lower()
        if level == "generating":
            self.num_generating += 1
        elif level == "terminal":
            self.num_terminal += 1
        elif level == "primary":
            self.num_primary += 1
        elif level == "secondary":
            self.num_secondary += 1
        elif level == "tertiary":
            self.num_tertiary += 1
        elif level == "quaternary":
            self.num_quaternary += 1
        
        self.num_entry += 1
        self.entries[level].append(coord)
        
        assert self.num_entry == self.num_generating + self.num_terminal + self.num_primary + self.num_secondary + self.num_tertiary + self.num_quaternary, "Incorrect number of junctions"
        
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
        self.num_generating -= 1
        
        
def test():
    junctions = Junctions()
    junctions.add(coord=(123, 345), level="Primary")
    assert junctions.num_entry == 1
    assert junctions.num_primary == 1
    junctions.add(coord=(110, 500), level="generating")
    junctions.add(coord=(500, 500), level="generating")
    assert junctions.num_entry == 3
    assert junctions.num_generating == 2
    junctions.remove_end_generating()
    assert junctions.num_entry == 2
    assert junctions.num_generating == 1
    assert len(junctions.entries["generating"]) == 1
    print("All tests passed")


if __name__ == "__main__":
    test()
    