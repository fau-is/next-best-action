# coding=utf-8
"""
This module contains the DCR graph representation. It also contains the XML format parsing functionality
"""
import xml.etree.ElementTree as Etree

from dcr_activity import DCRActivityBase, DCRActivityNest, DCRActivity
from dcr_conn import DCRConnection, ConnectionTypes, Condition, Milestone



class DCRGraph(object):
    """
    The representation of the DCR graph
    """
    # Variable for global access to the one instance of the dcr_graph
    __dcr_graph = None

    def __init__(self, xml_path):
        """Constructor for the DCR Graph"""
        # Set up instance variables
        self.Connections = []
        self.Mappings = {}
        self.Nodes = []
        self.ConditionTargets = []
        self.MilestoneTargets = []
        self.InitialIncluded = []
        self.InitialPending = []
        self.InitialExecuted = []

        # Init XML reader for class
        self.dcr_xml = Etree.parse(xml_path)
        self.dcr_xml_root = self.dcr_xml.getroot()

        # Start parsing the DCR Graph xml
        self.create_label_mapping()
        self.create_activities_dcr_graph()
        self.create_connections_dcr_graph()
        self.create_initial_marking()
        self.set_condition_targets()
        self.set_milestone_targets()

        # Put the graph into the class static variable
        DCRGraph.__dcr_graph = self

    def create_label_mapping(self):
        """
        In DCR-Graph xml the activity label and id have to be matched to relate them
        :return: a list with related mappings
        """
        mappings = {}
        for mapping in self.dcr_xml_root.iter('labelMapping'):
            mappings[mapping.get('eventId')] = mapping.get('labelId')
        self.Mappings = mappings

    def __add_event_to_graph(self, event):
        """
        Adds an event to the graph
        :param event: The event to be added
        """
        roles = []
        for role in event.iter('role'):
            if role.text is not None:
                roles.append(role.text)
        event_id: str = event.get('id')
        event_name = self.Mappings.get(event_id)
        node = DCRActivity(event_id, event_name)
        node.set_roles(roles)
        self.add_node(node)

    def __add_nest_to_graph(self, nest):
        """
        Adds a nested activity to the DCR Graph
        :param nest: The root of the nested activity
        """
        sub_events = []
        activities = []
        roles = []
        first = True
        for event in nest.iter('event'):
            if first:
                # Since the naming of a nested activity contains event itself the nesting activity
                # would be duplicated, we do not want this to happen, therefore the first event the nesting
                # activity is skipped.
                first = False
                continue
            self.__add_event_to_graph(event)
            self.get_node(event.get('id'))
            sub_events.append(event.get('id'))
        event_id: str = nest.get('id')
        event_name = self.Mappings.get(event_id)
        nest_custom = nest.find('custom')
        for role in nest_custom.iter('role'):
            roles.append(role.text)
        for activity in sub_events:
            activities.append(self.get_node(activity))
        nest_node = DCRActivityNest(event_id, event_name, activities)
        for activity in nest_node.Activities:
            activity.set_nesting_activity(nest_node)
        nest_node.set_roles(roles)
        self.add_node(nest_node)

    def create_activities_dcr_graph(self):
        """
        Only purpose is to get the activities from the DCR Graph XML into the data structure
        :return: None
        """
        for events in self.dcr_xml_root.iter('events'):
            for event in events:
                event_type = event.get('type')
                if event_type == 'nesting':
                    self.__add_nest_to_graph(event)
                else:
                    self.__add_event_to_graph(event)

    def create_initial_marking(self):
        """
        Sets the initial state of the DCR_Graph, which activities are included from the beginning
        """
        for include in self.dcr_xml_root.iter('included'):
            for event in include:
                event_id = event.get('id')
                node: DCRActivityBase = self.get_node(event_id)
                self.InitialIncluded.append(node)
        for executed_xml in self.dcr_xml_root.iter('executed'):
            for event in executed_xml:
                event_id = event.get('id')
                node: DCRActivityBase = self.get_node(event_id)
                self.InitialExecuted.append(node)
        for pendingResponse in self.dcr_xml_root.iter('pendingResponses'):
            for event in pendingResponse:
                event_id = event.get('id')
                node: DCRActivityBase = self.get_node(event_id)
                self.InitialPending.append(node)

    def create_connections_dcr_graph(self):
        """
        Creates all constraints (connections) of a DCR Graph from the XML
        :return: None
        """
        for constraint in self.dcr_xml_root.iter('constraints'):
            for constraint_type in constraint:
                for connection in constraint_type:
                    connection_source = connection.get('sourceId')
                    connection_destination = connection.get('targetId')
                    source_node = self.get_node(connection_source)
                    target_node = self.get_node(connection_destination)
                    connection_type = ConnectionTypes[connection.tag]
                    dcr_connection = DCRConnection.create_connection(source_node, target_node, connection_type)
                    self.add_connection(dcr_connection)

    def get_node(self, node_id: str) -> DCRActivityBase:
        """
        Returns a certain node, using the activity Id to locate it
        :param node_id:The node id which is located
        :return: A DCRActivityBase object
        """
        for node in self.Nodes:
            if node.ActivityId == node_id:
                return node

    def get_node_by_name(self, activity_name):
        """
        Gets a node by using the
        :param activity_name:
        :return:
        """
        for activity_id, event_name in self.Mappings.items():
            if event_name == activity_name:
                return self.get_node(activity_id)

    def add_node(self, node: DCRActivityBase):
        """
        Adds a node to the list of
        :type node: DCRNode
        :param node: The Node that will be added to the Graphs list
        """
        self.Nodes.append(node)

    def remove_node(self, node: DCRActivityBase):
        """
        Removes a node from the DCR Graph
        :param node: Node to be removed
        :return: None
        """
        self.Nodes.remove(node)

    def add_role(self, role):
        """
        Adds a role to the graph
        :param role: the role to be added
        :return: void
        TODO
        """
        raise NotImplementedError("Function still under construction")
        # self.Roles.append(role)

    def add_connection(self, connection: DCRConnection):
        """
        Adds a connection to the DCR Graph
        :param connection: Connection of type
        :return:
        """
        self.Connections.append(connection)

    def remove_connection(self, connection):
        """
        Removes a connection from tha graph
        :param connection: the connection to be removed
        :return: None
        """
        self.Connections.remove(connection)

    @staticmethod
    def get_graph_instance(xml_path=None):
        """
        Gets the instance of the graph
        :return: graph instance
        """
        if xml_path is None and DCRGraph.__dcr_graph is not None:
            return DCRGraph.__dcr_graph
        elif xml_path is not None:
            return DCRGraph(xml_path)
        else:
            raise TypeError('DCR graph has not been initialized yet but no File parameter was given')

    def get_connections_outgoing(self, source_id):
        """
        Gets all connections that start at the node related to the source id
        :param source_id: the id of the connection start node
        :return: list of connections starting at the node
        """
        node = self.get_node(source_id)
        connections_outgoing = []
        for connection in self.Connections:
            if connection.StartNode is node:
                connections_outgoing.append(connection)
        return connections_outgoing

    def get_connections_incoming(self, target_id, connection_type=None):
        """
        Gets all connections that start at the node related to the source id
        :param connection_type:
        :param target_id: the id of the connection start node
        :return: list of connections starting at the node
        """
        node = self.get_node(target_id)
        connections_incoming = []
        for connection in self.Connections:
            if connection.EndNode is node:
                if connection_type is None:
                    connections_incoming.append(connection)
                else:
                    if isinstance(connection, connection_type):
                        connections_incoming.append(connection)
        return connections_incoming

    def set_condition_targets(self):
        """
        Sets all nodes into a list that are condition targets
        :return:
        """
        conditions = []
        for connection in self.Connections:
            if type(connection) is Condition:
                conditions.append(connection)
        targets = []
        for condition in conditions:
            if condition not in targets:
                targets.append(condition.EndNode)
        self.ConditionTargets = targets

    def get_condition_targets(self):
        """
        Get all condition target activities from the graph
        :return [DCRActivityBase] where each activity is a condition target
        """
        return self.ConditionTargets

    def set_milestone_targets(self):
        """
        Sets the list of activities for a DCR graphs process model that are
        """
        milestones = []
        for connection in self.Connections:
            if isinstance(connection, Milestone):
                milestones.append(connection)
        targets = []
        for milestone in milestones:
            if milestone not in targets:
                targets.append(milestone.EndNode)
        self.MilestoneTargets = targets
