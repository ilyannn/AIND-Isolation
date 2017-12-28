import random
import math


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def free_moves_score(game, player, depth, unroll_moves=3, discount_opponent=1):

    choices = game.get_legal_moves()

    if not choices:
        return game.utility(player)

    sign = 1 if game.active_player == player else -1
    player_free_score = sign * float(len(choices))

    if depth <= 1:
        return player_free_score

    follow_choices = random.sample(choices, min(unroll_moves, len(choices)))
    opponent_scores = map(lambda choice: free_moves_score(game.forecast_move(choice), player, depth-1, 1), follow_choices)

    return player_free_score + discount_opponent * sum(opponent_scores) / len(follow_choices)


def valid_moves(board, loc):
    """Generate the list of possible moves for an L-shaped motion (like a
    knight in chess).
    """
    if loc == None:
        return board.get_blank_spaces()

    r, c = loc
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
              (1, -2), (1, 2), (2, -1), (2, 1)]
    valid_moves = [(r + dr, c + dc) for dr, dc in directions
               if board.move_is_legal((r + dr, c + dc))]
    random.shuffle(valid_moves)
    return valid_moves


def cell_distance(game, player):
    """Find distance from the player for all empty cells on the board.

    :param game:   board to examine
    :param player: player from whose position we classify
    :return:       a dictionary of pairs (cell, distance)
    """
    step = 0
    visited = set()
    distance = dict()

    new_visited = set()
    new_visited.add(game.get_player_location(player))

    while new_visited:

        for loc in new_visited:
            distance[loc] = step

        visited.update(new_visited)
        step += 1

        for loc in new_visited.copy():
            new_visited.update(valid_moves(game, loc))

        new_visited.difference_update(visited)

    return distance


def use(x, y):
    """A plain math sigmoid function.

    Parameters
    ----------
    x : float
        An input value

    Returns
    -------
    float
        The sigmoid function value, from -1 to +1.

    """

    return math.atan(x-y)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    choices = game.get_legal_moves()

    if not choices:
        return game.utility(player)

    for_player = cell_distance(game, player)
    for_opponent = cell_distance(game, game.get_opponent(player))

    def usefulness(cell):
        p = for_player.get(cell, None)
        o = for_opponent.get(cell, None)

        if p is None and o is None:
            return 0
        if o is None:
            return 1
        if p is None:
            return -1

        return use(o, p)

    cells = game.get_blank_spaces()
    return sum(usefulness(cell) for cell in cells)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    return free_moves_score(game, player, 2, 3, 1)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return free_moves_score(game, player, 2, 3, 0.8)


def custom_score_4(game, player):
    return free_moves_score(game, player, 2, 3, 1.2)


def custom_score_5(game, player):
    return free_moves_score(game, player, 2, 3, 1.4)


def custom_score_6(game, player):
    return free_moves_score(game, player, 2, 10, 1)



class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        def min_or_max_value(board, mdepth, is_max):
            """Find the min/max value and the choice leading to that value.

            :param board: position to look into
            :param mdepth: maximum search depth (0 returns immediately)
            :param is_max: True if we want the answer maximizing value
            :return: pair (resulting value, choice)
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            choices = board.get_legal_moves(board.active_player)

            if not choices:
                return board.utility(game.active_player), (-1, -1)

            if 0 == mdepth:
                return self.score(board, game.active_player), (-1, -1)

            def applied(choice):
                """Apply the move to the board and return result.
                """
                computed = min_or_max_value(board.forecast_move(choice), mdepth - 1, not is_max)
                return computed[0], choice

            values = map(applied, choices)
            return (max if is_max else min)(values)

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return min_or_max_value(game, depth, True)[1]


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        if game.get_legal_moves():

            depth = 1
            try:
                while True:
                    # The try/except block will automatically catch the exception
                    # raised when the timer is about to expire.
                    best_move = self.alphabeta(game, depth)
                    depth += 1

            except SearchTimeout:
                pass

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        def min_or_max_alphabeta(board, current_alpha, current_beta, mdepth):
            """Find the min/max value and the choice leading to that value.

            :param board:          position to look into
            :param current_alpha:  limits the lower bound of search on minimizing layers
            :param current_beta:   limits the upper bound of search on maximizing layers
            :param mdepth:         maximum search depth (0 returns immediately)
            :return:               pair (resulting value, choice)
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            choices = board.get_legal_moves()

            if not choices:
                return board.utility(game.active_player), (-1, -1)

            if mdepth <= 0:
                return self.score(board, game.active_player), choices[0]

            is_max = game.active_player == board.active_player # are we maximising?
            best_score = None   # will be filled because choices is non-empty
            best_choice = None

            for choice in choices:

                choice_score, _ = min_or_max_alphabeta(board.forecast_move(choice), current_alpha, current_beta, mdepth - 1)

                if (best_score is None
                        or (is_max and choice_score > best_score)
                        or (not is_max and choice_score < best_score)):
                    best_score = choice_score
                    best_choice = choice

                if is_max and best_score >= current_beta or not is_max and best_score <= current_alpha:
                    break

                if is_max and best_score > current_alpha:
                    current_alpha = best_score

                if not is_max and best_score < current_beta:
                    current_beta = best_score

            return best_score, best_choice

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return min_or_max_alphabeta(game, alpha, beta, depth)[1]

    def __str__(self):
        return "AlphaBetaPlayer (" + str(self.score.__name__) + ")"