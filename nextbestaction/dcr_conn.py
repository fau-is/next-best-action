# coding=utf-8
"""The module contains all classes for the representations of DCR relations"""
from abc import ABC, abstractmethod
from enum import Enum, auto


class ConnectionTypes(Enum):
    """
    Condition: creates a relation between an activity A and
    an activity B such that B can only occur if first A has occurred

    Response: The Response connection creates a relation between an activity A and an activity B such that B has to
    occur, at least once, at some point after, if A occurs. B can occur even if A never occurs. But if A, then B.

    Include: creates a relation between an activity A and an activity B such that the occurrence of activity A
    makes possible the occurrence of activity B if it wasn't previously included in the workflow

    Exclude: creates a relation between an activity A and an activity B such that B cannot occur
    if first A has occurred.

    Milestone: creates a relation between an activity A and an activity B such that B can occur initially.
    But if A becomes pending for a response connection by another activity C, then B cannot occur
    until A has occurred

    Spawn: The Spawn connection creates a relation between an activity A and a sub-activity B such that,
    when A occurs, a new instance of B is created
    """
    condition = auto()
    response = auto()
    include = auto()
    exclude = auto()
    milestone = auto()


class DCRConnection(ABC):
    """
    Abstract class for DCR relations to be inherited
    """

    def __init__(self, start_node, end_node):
        self.StartNode = start_node
        self.EndNode = end_node


    @staticmethod
    def create_connection(start_node, end_node, connection_type: ConnectionTypes):
        """
        Static method that builds a certain connection based on the enum connection type
        :param expression: If expression exists default is no expression
        :param start_node: The origin node of the connection
        :param end_node: destination node of the connection
        :param connection_type: the type of constraint related to the connection
        :return: Instance of the constraint
        """
        connection = None
        if connection_type == ConnectionTypes.condition:
            connection = Condition(start_node, end_node)
        elif connection_type == ConnectionTypes.response:
            connection = Response(start_node, end_node)
        elif connection_type == ConnectionTypes.exclude:
            connection = Exclude(start_node, end_node)
        elif connection_type == ConnectionTypes.include:
            connection = Include(start_node, end_node)
        elif connection_type == ConnectionTypes.milestone:
            connection = Milestone(start_node, end_node)
        if connection is not None:
            return connection
        else:
            raise ValueError('connection was None')

    @abstractmethod
    def perform_transition(self, marking):
        """
        Distributor for the performance of marking transition
        :param marking:
        :return: True if conformant, False if not
        """
        pass


class Include(DCRConnection):
    """
    The representation of an Include relations
    :DCRConnection implementer
    """

    def __init__(self, start_node, end_node):
        """
        Constructor for an Include connection
        :super DCRConnection
        :param start_node
        :param end_node
        :see also DCRConneciton
        """
        super().__init__(start_node, end_node)

    def perform_transition(self, marking):
        """
        Includes the target activities
        :param marking: Needed to manipulate the marking
        :return:
        """
        if self.EndNode not in marking.Included:
            marking.Included.append(self.EndNode)
        if hasattr(self.EndNode, "Activities"):
            for activity in self.EndNode.Activities:
                if activity not in marking.Included:
                    marking.Included.append(activity)
        elif self.EndNode.NestingActivity is not None:
            if self.EndNode.NestingActivity is not marking.Included:
                marking.Included.append(self.EndNode.NestingActivity)


class Milestone(DCRConnection):
    """
    The class that represents a Milestone relation
    """

    def __init__(self, start_node, end_node):
        """
        Constructor for the Milestone
        :param start_node:
        :param end_node:
        """
        super().__init__(start_node, end_node)
        self.EndNode.set_is_milestone_target()

    def perform_transition(self, marking):
        """
        Empty method stub for Milestone and Condition
        :param marking:
        :return:
        """
        pass


class Condition(DCRConnection):
    """
    The class that represents a condition relation
    """

    def __init__(self, start_node, end_node):
        super().__init__(start_node, end_node)
        self.EndNode.set_is_condition_target()

    def perform_transition(self, marking):
        """
        Empty method stub in condition and milestone
        :param marking:
        :return:
        """
        pass


class Exclude(DCRConnection):
    """
    The class represents the Exclude relation
    """

    def __init__(self, start_node, end_node):
        super().__init__(start_node, end_node)

    def perform_transition(self, marking):
        """
          Override for base method
          :param marking: Base-Marking on which the transition is checked
          :return: True if conformant, False if not
          """
        if self.EndNode in marking.Included:
            marking.Included.remove(self.EndNode)
        if hasattr(self.EndNode, "Activities"):
            for activity in self.EndNode.Activities:
                if activity in marking.Included:
                    marking.Included.remove(activity)
        elif self.EndNode.NestingActivity is not None:
            if not any(activity in marking.Included for activity in self.EndNode.NestingActivity.Activities):
                marking.Included.remove(self.EndNode.NestingActivity)


class Response(DCRConnection):
    """
    The class represents a Response connection in a DCR graph
    """

    def __init__(self, start_node, end_node):
        """
        Constructor for Response, calls super __init__
        :param start_node:
        :param end_node:
        """
        super().__init__(start_node, end_node)

    def perform_transition(self, marking):
        """
        Performs the transition for a response connection
        :param marking:
        :return:
        """
        if self.EndNode not in marking.PendingResponse:
            marking.PendingResponse.append(self.EndNode)

        if hasattr(self.EndNode, "Activities"):
            for activity in self.EndNode.Activities:
                if activity not in marking.PendingResponse:
                    marking.PendingResponse.append(activity)
        elif self.EndNode.NestingActivity is not None:
            # if any(activity in self.EndNode.NestingActivity.Activities for activity in marking.PendingResponse):
            if self.EndNode.NestingActivity not in marking.PendingResponse:
                marking.PendingResponse.append(self.EndNode.NestingActivity)
