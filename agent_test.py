"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent

from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        pass

    def test_math(self):
        self.assertAlmostEqual(game_agent.usefulness_sigmoid(100, 0), 1, places=2)
        self.assertAlmostEqual(game_agent.usefulness_sigmoid(0, 100), -1, places=2)
        self.assertAlmostEqual(game_agent.usefulness_atan(100, 0), 1, places=2)
        self.assertAlmostEqual(game_agent.usefulness_atan(0, 100), -1, places=2)
        self.assertEqual(game_agent.usefulness_sigmoid(100, 100), 0)
        self.assertEqual(game_agent.usefulness_sigmoid(100, 100), 0)


if __name__ == '__main__':
    unittest.main()
