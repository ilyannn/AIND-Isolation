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
        reload(game_agent)
        self.player1 = "Player1"
        self.player2 = "Player2"
        self.game = isolation.Board(self.player1, self.player2)

    def testEmpty(self):
        self.assertEqual(game_agent.custom_score(self.game, self.game.active_player), 0)

    def testSplit(self):
        for y in range(0, self.game.height):
            for x in range(2, 3):
                self.game.apply_move([x, y])
        self.assertEqual(game_agent.custom_score(self.game, self.game.active_player), (self.game.width - 4)*self.game.height)


if __name__ == '__main__':
    unittest.main()
