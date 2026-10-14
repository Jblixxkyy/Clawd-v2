import sys
import os
os.chdir("D:\clawd\Clawd-v2")
sys.path.insert(0, ".")

from tests.test_web_app import WebAppTests
import unittest

# Just run a simple test to see if things work
suite = unittest.TestLoader().loadTestsFromTestCase(WebAppTests)
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
print(f"Tests run: {result.testsRun}, Failures: {len(result.failures)}, Errors: {len(result.errors)}")