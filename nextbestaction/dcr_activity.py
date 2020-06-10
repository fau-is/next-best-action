# coding=utf-8
""" Contains all DCR activity classes and their definitions"""

from abc import ABC


class DCRActivityBase(ABC):
    """
    The base class for all DCR activities
    """

    def __init__(self, activity_id=str, activity_name=str):
        """
        The constructor of the abstracted super class DCRActivityBase that defines how attributes
        have to look like
        :param activity_id: The activity id
        :param activity_name: The name of the activity
        :param attributes: attributes that
        """
        if activity_id is None or activity_name is None:
            raise TypeError('Parameters may not be null')
        self.ActivityId = activity_id
        self.ActivityName = activity_name
        self.Roles = []
        self.IsConditionTarget = False
        self.IsMilestoneTarget = False

    def set_roles(self, roles: []):
        """
        Sets the roles for the activity
        :param roles: roles that are set for the activity
        """
        for role in roles:
            self.Roles.append(role)

    def set_is_condition_target(self):
        """
        Set flag to indicate if the node is a condition target
        """
        self.IsConditionTarget = True

    def set_is_milestone_target(self):
        """
        Set flag to indicate if the node is a milestone target
        """
        self.IsMilestoneTarget = True


class DCRActivity(DCRActivityBase):
    """A default DCRActivity"""

    def __init__(self, activity_id, activity_name):
        """
        This method constructs an activity based on certain events. Subclass of DCRActivityBase
        :param activity_id: The id of the event tag
        :param activity_name: the activity name of the mappings tag in the xml
        """
        self.NestingActivity = None
        super().__init__(activity_id, activity_name)

    def set_nesting_activity(self, nesting_activity):
        """
        Set the nesting activity that contains this activity
        :param nesting_activity: DCRActivityNest
        """
        self.NestingActivity = nesting_activity


class DCRActivityNest(DCRActivityBase):
    """Representation of Nesting activity of DCR graphs"""

    def __init__(self, activity_id=str, activity_name=str, activities=None):
        """
        Constructor for a nesting activity
        :param activity_id: The id of the event tag
        :param activity_name: the activity name of the mappings tag in the xml
        :param activities: Contains all activities that are nested under the nesting activity
        """
        if activities is None:
            activities = []
        self.Activities = activities
        super().__init__(activity_id, activity_name)
