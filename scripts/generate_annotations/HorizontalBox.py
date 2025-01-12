class HorizontalBox:
    """hbb manager for an image"""

    def __init__(self, junctions: list = None, branches: list = None) -> None:
        self.junctions = junctions
        self.branches = branches
        self.rects_junctions = list()
        self.rects_branches = list()

    def run_junctions(self, width, height) -> list:
        for x, y in self.junctions:
            pt1 = (x - width // 2, y - height // 2)
            pt2 = (x + width // 2, y + height // 2)
            self.rects_junctions.append(tuple([pt1, pt2]))
        return self.rects_junctions

    def run_branches(self) -> list:
        for x1, y1, x2, y2 in self.branches:
            # pt1 = (x1, y1): end point         |   secondary junction  | junction
            # pt2 = (x2, y2): primary junction  |   end point           | end point
            self.rects_branches.append(tuple([x1, y1, x2, y2]))
        return self.rects_branches
