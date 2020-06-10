# coding=utf-8
"""
The module implements the Marking of a DCR Graph
"""
import dcr_conn

from dcr_activity import DCRActivityBase
from dcr_graph import DCRGraph


class Marking(object):
    """
    This class is meant to represent a possible marking of a Graph
        A graph marking contains three distinct sets of states for each activity
        1. Included:
            - The event is ready to be executed
        2. Pending Response:
            - The activity needs to be executed because of the response constraint
        3. Executed:
            - The event has been executed
        4. Excluded:
        If an activity is not included it is excluded
    """
    InitialMarking = None

    @staticmethod
    def get_initial_marking():
        """
        Method creates a copy of the Initial Marking of the graph
        :return: A copy of the Initial Marking of a graph
        """
        dcr_graph = DCRGraph.get_graph_instance()
        initial_included = []
        for incl in dcr_graph.InitialIncluded:
            initial_included.append(incl)
        initial_pending = []
        for pending in dcr_graph.InitialPending:
            initial_pending.append(pending)
        initial_executed = []
        for exec in dcr_graph.InitialExecuted:
            initial_executed.append(exec)
        return Marking(initial_included, initial_pending,
                       initial_executed)

    def __init__(self, included: [DCRActivityBase], pending_response: [DCRActivityBase], executed: [DCRActivityBase]):
        self.Included = included
        self.PendingResponse = pending_response
        self.Executed = executed
        self.dcr_graph: DCRGraph = DCRGraph.get_graph_instance()

    def perform_transition_connection(self, connection):
        """
        Performs a transition for each connection
        :return: new Marking
        """
        return connection.perform_transition(self)

    def perform_transition_node(self, node):
        """
        Performs a transition on a marking and a connection
        :param trace_data:
        :param event: Event from the trace
        :param node: node for what connections are enabled
        :return:
        """
        if node is None:
            return True  # Global variable

        if self.node_is_blocked(node):
            return False

        if node.NestingActivity is not None:
            if self.node_is_blocked(node.NestingActivity):
                return False

        if node not in self.Included:
            return False

        if node not in self.Executed:
            self.Executed.append(node)
            if node.NestingActivity is not None:
                included_activities = self.get_included_activities(node.NestingActivity.Activities)
                if set(included_activities).issubset(set(self.Executed)):
                    self.Executed.append(node.NestingActivity)

        if node in self.PendingResponse:
            self.PendingResponse.remove(node)
            if node.NestingActivity is not None:
                pending = any(activity in node.NestingActivity.Activities for activity in self.PendingResponse)
                if not pending and node.NestingActivity in self.PendingResponse:
                    self.PendingResponse.remove(node.NestingActivity)

        connections = self.dcr_graph.get_connections_outgoing(node.ActivityId)
        if node.NestingActivity is not None:
            connections.extend(self.dcr_graph.get_connections_outgoing(node.NestingActivity.ActivityId))
        for connection in connections:
            self.perform_transition_connection(connection)
        return True

    def node_is_blocked(self, node):
        """
        Checks if the node is blocked in the current constellation of the marking
        :param trace_data: The whole related trace data
        :param node: The node to be checked
        :return: True if blocked, False if not
        """
        blocked = False
        # Check only if the node is a milestone target
        if node.IsMilestoneTarget:
            milestones = self.dcr_graph.get_connections_incoming(node.ActivityId, dcr_conn.Milestone)
            for milestone in milestones:
                # If the Start Node is included in the model as well as pending
                # the target is blocked
                if milestone.StartNode in self.Included and \
                        milestone.StartNode in self.PendingResponse:
                    blocked = True

        # Check only if the node is a condition target
        if node.IsConditionTarget:
            conditions = self.dcr_graph.get_connections_incoming(node.ActivityId, dcr_conn.Condition)
            for condition in conditions:
                # If the Start Node connection is included in the model
                # as well as not executed yet the activity is blocked
                if condition.StartNode in self.Included \
                        and condition.StartNode not in self.Executed:
                    blocked = True
        return blocked

    def get_included_activities(self, activities: []):
        """
        Get all included activities in a list
        :param activities: The list of activities to be checked
        :return: the list of included nested activities
        """
        included_nested_activities = [e for e in activities if e in self.Included]
        return included_nested_activities
