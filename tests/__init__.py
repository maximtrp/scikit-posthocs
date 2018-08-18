import unittest
import tests.test_posthocs


def posthocs_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_posthocs)
    return suite
