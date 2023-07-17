from abc import ABC


class AbstractCurve(ABC):
    def __init__(self):
        pass

    def length(self):
        pass


class Line(AbstractCurve):
    pass


class BezierCurve(AbstractCurve):
    pass
