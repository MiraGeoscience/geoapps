import ipywidgets as widgets


class Widget:
    """
    Base class for geoapps.Widget
    """

    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            if getattr(self, "_" + key, None) is not None:
                try:
                    getattr(self, "_" + key).value = value
                except:
                    pass

    def widget(self):
        ...
