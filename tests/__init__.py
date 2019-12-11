import unittest
import tests.test_posthocs

def posthocs_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(tests.test_posthocs)
    return suite
