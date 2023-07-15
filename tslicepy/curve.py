from abc import ABC


class AbstractCurve(ABC):
    def __init__(self):
        pass


class Line(AbstractCurve):
    pass


class BezierCurve(AbstractCurve):
    pass
