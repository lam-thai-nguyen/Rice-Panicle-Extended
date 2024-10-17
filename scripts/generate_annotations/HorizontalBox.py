class HorizontalBox:
    def __init__(self, junctions: list) -> None:
        self.junctions = junctions
        self.rects = list()

    def run(self, width, height) -> list:
        for x, y in self.junctions:
            pt1 = (x - width // 2, y - height // 2)
            pt2 = (x + width // 2, y + height // 2)
            self.rects.append(tuple([pt1, pt2]))
        return self.rects
    