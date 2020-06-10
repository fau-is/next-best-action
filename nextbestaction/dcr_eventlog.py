# coding=utf-8
"""The module implements a data structure to represent an event log"""


class Event(object):
    """
    The event class that represents an atomic event
    """

    def __init__(self, event_name):
        """
        Default constructor of the Event class
        """
        self.EventName = str(event_name)


class Trace(object):
    """
    Represents a trace of a log
    """

    def __init__(self, new_instance):
        """
        Default constructor of a Trace, a trace is a set of related events often also called a case
        a case is the instance of one process at a time
        """
        self.Events = []

        for event in new_instance:
            self.Events.append(Event(event))

