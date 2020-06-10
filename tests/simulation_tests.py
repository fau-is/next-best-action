import unittest
from dcr_graph import DCRGraph
from dcr_marking import Marking


def checkCandidate(new_instance):

    dcr = DCRGraph("./test_Resources/dcr.xml")
    marking = Marking.get_initial_marking()

    for event in new_instance:
        node = dcr.get_node_by_name(str(event))
        if not marking.perform_transition_node(node):
            return False

    if len(marking.PendingResponse) != 0:
        for pending in marking.PendingResponse:
            if pending in marking.Included:
                return False

    return True


class Import_model(unittest.TestCase):
    def test_import(self):
        self.assertEqual(checkCandidate([1, 2, 3]), True)
        self.assertEqual(checkCandidate([1, 3]), False)
        self.assertEqual(checkCandidate([1, 1]), False)
        self.assertEqual(checkCandidate([3]), False)


if __name__ == '__main__':
    unittest.main()
