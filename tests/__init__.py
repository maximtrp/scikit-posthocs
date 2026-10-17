import unittest


def posthocs_suite():
    from tests import test_posthocs

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_posthocs)
    return suite
