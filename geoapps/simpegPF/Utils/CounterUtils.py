from six import string_types
import time
import numpy as np
from functools import wraps


class Counter:
    """
        Counter allows anything that calls it to record iterations and
        timings in a simple way.

        Also has plotting functions that allow quick recalls of data.

        If you want to use this, import *count* or *timeIt* and use them as
        decorators on class methods.

        ::

            class MyClass(object):
                def __init__(self, url):
                    self.counter = Counter()

                @count
                def MyMethod(self):
                    pass

                @timeIt
                def MySecondMethod(self):
                    pass

            c = MyClass('blah')
            for i in range(100): c.MyMethod()
            for i in range(300): c.MySecondMethod()
            c.counter.summary()

    """

    def __init__(self):
        self._countList = {}
        self._timeList = {}

    def count(self, prop):
        """
            Increases the count of the property.
        """
        assert isinstance(prop, str), "The property must be a string."
        if prop not in self._countList:
            self._countList[prop] = 0
        self._countList[prop] += 1

    def countTic(self, prop):
        """
            Times a property call, this is the init call.
        """
        assert isinstance(prop, str), "The property must be a string."
        if prop not in self._timeList:
            self._timeList[prop] = []
        self._timeList[prop].append(-time.time())

    def countToc(self, prop):
        """
            Times a property call, this is the end call.
        """
        assert isinstance(prop, str), "The property must be a string."
        assert prop in self._timeList, "The property must already be in the dictionary."
        self._timeList[prop][-1] += time.time()

    def summary(self):
        """
            Provides a text summary of the current counters and timers.
        """
        print("Counters:")
        for prop in sorted(self._countList):
            print("  {:<40}: {:8d}".format(prop, self._countList[prop]))
        print("\nTimes:" + " " * 40 + "mean      sum")
        for prop in sorted(self._timeList):
            l = len(self._timeList[prop])
            a = np.array(self._timeList[prop])
            print(f"  {prop:<40}: {a.mean():4.2e}, {a.sum():4.2e}, {l:4d}x")


def count(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        counter = getattr(self, "counter", None)
        if type(counter) is Counter:
            counter.count(self.__class__.__name__ + "." + f.__name__)
        out = f(self, *args, **kwargs)
        return out

    return wrapper


def timeIt(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        counter = getattr(self, "counter", None)
        if type(counter) is Counter:
            counter.countTic(self.__class__.__name__ + "." + f.__name__)
        out = f(self, *args, **kwargs)
        if type(counter) is Counter:
            counter.countToc(self.__class__.__name__ + "." + f.__name__)
        return out

    return wrapper
