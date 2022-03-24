# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    ####################################################################################################################
    # Create a variable to store our current state
    start_state = problem.getStartState()  # we begin our search in the start state

    start_path = []

    source_node = (start_state, start_path)

    # Create a stack data structure to store the nodes we want to search from
    dfs_stack = util.Stack()

    # Create a set to store the states we have visited
    visited_states = set()  # our set is initially empty because we have not searched from any states yet

    # Begin search by pushing the source node to the stack
    dfs_stack.push(source_node)  # source node contains our start state

    while dfs_stack.isEmpty() != 1:

        # Pop a search node from the top of the stack
        search_node = dfs_stack.pop()

        # Store the state and the path from the search node
        current_state = search_node[0]
        current_path = search_node[1]

        if problem.isGoalState(current_state):
            return current_path

        if current_state not in visited_states:
            visited_states.add(current_state)

        list_of_successors = problem.getSuccessors(current_state)

        for successor in list_of_successors:

            successor_state = successor[0]
            successor_path = [successor[1]]
            total_path = current_path + successor_path
            successor_node = (successor_state, total_path)

            if successor_state not in visited_states:
                dfs_stack.push(successor_node)
    ####################################################################################################################
    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    ####################################################################################################################
    # Create a variable to store our current state
    start_state = problem.getStartState()  # we begin our search in the start state

    start_path = []

    source_node = (start_state, start_path)

    # Create a Queue data structure to store the nodes we want to search from
    bfs_queue = util.Queue()

    # Create a set to store the states we have visited
    visited_states = set()  # our set is initially empty because we have not searched from any states yet

    # Begin search by pushing the source node to the Queue
    bfs_queue.push(source_node)  # source node contains our start state

    visited_states.add(start_state)

    while bfs_queue.isEmpty() != 1:

        # Pop a search node from the top of the queue
        search_node = bfs_queue.pop()

        # Store the state and the path from the search node
        current_state = search_node[0]
        current_path = search_node[1]

        if problem.isGoalState(current_state):
            return current_path

        list_of_successors = problem.getSuccessors(current_state)

        for successor in list_of_successors:

            successor_state = successor[0]
            successor_path = [successor[1]]
            total_path = current_path + successor_path
            successor_node = (successor_state, total_path)

            if successor_state not in visited_states:
                bfs_queue.push(successor_node)
                visited_states.add(successor_state)
    ####################################################################################################################
    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    ####################################################################################################################
    # Create a variable to store our current state
    start_state = problem.getStartState()  # we begin our search in the start state

    start_path = []

    start_step_cost = 0

    source_node = (start_state, start_path, start_step_cost)

    # Create a Priority Queue data structure to store the nodes we want to search from
    ucs_priority_queue = util.PriorityQueue()

    # Create a set to store the states we have visited
    visited_states = set()  # our set is initially empty because we have not searched from any states yet

    # Begin search by pushing the source node to the Priority Queue
    ucs_priority_queue.push(source_node, start_step_cost)  # source node contains our start state

    visited_states.add(start_state)

    current_shortest_cost = 0

    while ucs_priority_queue.isEmpty() != 1:

        # Pop a search node from the top of the Priority queue
        search_node = ucs_priority_queue.pop()

        # Store the state, path, and cost from the search node
        current_state = search_node[0]
        current_path = search_node[1]
        current_step_cost = search_node[2]

        if problem.isGoalState(current_state):
            return current_path

        list_of_successors = problem.getSuccessors(current_state)

        for successor in list_of_successors:

            successor_state = successor[0]
            successor_path = [successor[1]]
            successor_step_cost = successor[2]

            total_path = current_path + successor_path
            total_cost = current_step_cost + successor_step_cost

            successor_node = (successor_state, total_path, total_cost)

            if successor_state not in visited_states:
                ucs_priority_queue.push(successor_node, total_cost)
                visited_states.add(successor_state)
            if problem.isGoalState(successor_state):
                if current_shortest_cost == 0:
                    current_shortest_cost = total_cost
                if current_shortest_cost != 0:
                    if total_cost < current_shortest_cost:
                        ucs_priority_queue.push(successor_node, total_cost)
    ####################################################################################################################
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    ####################################################################################################################
    # Create a variable to store our current state
    start_state = problem.getStartState()  # we begin our search in the start state

    start_path = []

    start_step_cost = 0

    source_node = (start_state, start_path, start_step_cost)

    # Create a Priority Queue data structure to store the nodes we want to search from
    astar_priority_queue = util.PriorityQueue()

    # Create a set to store the states we have visited
    visited_states = set()  # our set is initially empty because we have not searched from any states yet

    # Begin search by pushing the source node to the Priority Queue
    astar_priority_queue.push(source_node, start_step_cost)  # source node contains our start state

    visited_states.add(start_state)

    current_shortest_cost = 0

    while astar_priority_queue.isEmpty() != 1:

        # Pop a search node from the top of the Priority queue
        search_node = astar_priority_queue.pop()

        # Store the state, path, and cost from the search node
        current_state = search_node[0]
        current_path = search_node[1]
        current_step_cost = search_node[2]

        if problem.isGoalState(current_state):
            return current_path

        list_of_successors = problem.getSuccessors(current_state)

        for successor in list_of_successors:

            successor_state = successor[0]
            successor_path = [successor[1]]
            successor_step_cost = successor[2]

            total_path = current_path + successor_path
            total_cost = current_step_cost + successor_step_cost
            heuristic_cost = heuristic(successor_state, problem)
            total_cost_heuristic = total_cost + heuristic_cost

            successor_node = (successor_state, total_path, total_cost)

            if successor_state not in visited_states:
                astar_priority_queue.push(successor_node, total_cost_heuristic)
                visited_states.add(successor_state)
            if problem.isGoalState(successor_state):
                if current_shortest_cost == 0:
                    current_shortest_cost = total_cost
                if current_shortest_cost != 0:
                    if total_cost < current_shortest_cost:
                        astar_priority_queue.push(successor_node, total_cost)
    ####################################################################################################################
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
