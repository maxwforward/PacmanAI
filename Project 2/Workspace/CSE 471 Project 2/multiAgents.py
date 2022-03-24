# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #################################################################################
        # Create a variable to store our current score
        current_score = 0

        # If the successor state is a win return a positive score
        if successorGameState.isWin():
            current_score = 9999999
            return current_score

        # If the successor state is a loss return a negative score
        if successorGameState.isLose():
            current_score = -9999999
            return current_score

        # Variables to store the total distances of scared and non-scared ghosts to pac-man
        total_distance_from_ghosts = 0
        total_distance_from_scared_ghosts = 0

        # For each ghost compute the manhattan distance and add the total amount of distances from Pac-Man
        for ghost in newGhostStates:
            ghost_position = ghost.getPosition()
            ghost_distance_from_pac_man = util.manhattanDistance(newPos, ghost_position)

            # If ghost is NOT scared
            if ghost.scaredTimer == 0:
                total_distance_from_ghosts = total_distance_from_ghosts + ghost_distance_from_pac_man
                # If Pac-Man is close to a ghost that is NOT scared, lower the score
                if ghost_distance_from_pac_man <= 1:
                    current_score = current_score - 1000

            # If ghost IS scared
            if ghost.scaredTimer != 0:
                total_distance_from_scared_ghosts = total_distance_from_scared_ghosts + ghost_distance_from_pac_man
                # If Pac-Man's new position contains a scared ghost, raise the score
                if ghost_distance_from_pac_man <= 1:
                    current_score = current_score + 100

        # Reward Pac-Man for keeping distance from the non-scared ghosts
        current_score = current_score + (total_distance_from_ghosts * 0.5)

        # Penalize Pac-Man for not going after scared ghosts
        current_score = current_score - total_distance_from_scared_ghosts

        # Get the current food from the game state
        current_food = currentGameState.getFood()

        # Create lists for the positions of foods
        successor_food_positions = newFood.asList()
        current_food_positions = current_food.asList()

        # Store the number of foods in the current and successor game states
        num_of_food_in_successor = len(successor_food_positions)
        num_of_food_in_current = len(current_food_positions)

        # Reward Pac-Man for consuming food
        if num_of_food_in_successor < num_of_food_in_current:
            current_score = current_score + 200

        # Variable to store the total distances of food to pac-man
        total_distance_from_food = 0

        # Variable to store length of closest food
        closest_food = -1

        # For each food compute the manhattan distance and find the closest food
        for food_position in successor_food_positions:
            food_distance_from_pac_man = util.manhattanDistance(newPos, food_position)
            total_distance_from_food = total_distance_from_food + food_distance_from_pac_man

            # Store distance is it is the closest food
            if closest_food < 0:
                closest_food = food_distance_from_pac_man
            else:
                if closest_food > food_distance_from_pac_man:
                    closest_food = food_distance_from_pac_man

        # Subtract the total distances from the score, penalizing Pac-Man for having a greater distance from the food
        current_score = current_score - total_distance_from_food

        # Subtract the distance of the closest food, penalizing Pac-Man for not moving towards the closest food
        if closest_food >= 0:
            current_score = current_score - (closest_food * 2)

        # Prevent Pac-Man from stopping
        if action == Directions.STOP:
            current_score = current_score - 50

        # Add the score to the successor state's current score and return it
        return successorGameState.getScore() + current_score
        #################################################################################


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #################################################################################
        evaluation_function = self.evaluationFunction
        possible_actions = gameState.getLegalActions()
        number_of_agents = gameState.getNumAgents()
        number_of_ghosts = (number_of_agents - 1)
        list_of_successors_pac_man = []
        scores_for_actions = []
        pac_man_agent_index = 0
        search_agent_depth = self.depth
        max_score = -999999999999999

        for action in possible_actions:
            list_of_successors_pac_man.append(gameState.generateSuccessor(pac_man_agent_index, action))

        # ==============================================================
        # Min-Max Helper Function
        # ==============================================================
        def min_max_helper(state, depth, agent_index):
            # Create variables to store information
            legal_actions_for_agent = state.getLegalActions(agent_index)
            list_of_successors = []
            current_score = 0

            # If game is over or we are done with recursion
            if depth == search_agent_depth or state.isWin() or state.isLose():
                current_score = evaluation_function(state)
                return current_score

            # If agent is Maximizing player (Pac-Man)
            # ************************************************
            if agent_index == pac_man_agent_index:
                current_score = -99999999999999999999
                # Generate list of child nodes
                for action in legal_actions_for_agent:
                    successor_node = state.generateSuccessor(agent_index, action)
                    list_of_successors.append(successor_node)
                # Search child nodes
                for successor in list_of_successors:
                    if agent_index == number_of_ghosts:
                        score = min_max_helper(successor, (depth + 1), agent_index)
                        if score > current_score:
                            current_score = score
                    else:
                        score = min_max_helper(successor, depth, (agent_index + 1))
                        if score > current_score:
                            current_score = score
                return current_score

            # If agent is Minimizing player (Ghost)
            # ************************************************
            else:
                current_score = 99999999999999999999
                # Generate list of child nodes
                for action in legal_actions_for_agent:
                    successor_node = state.generateSuccessor(agent_index, action)
                    list_of_successors.append(successor_node)
                # Search child nodes
                for successor in list_of_successors:
                    if agent_index == number_of_ghosts:
                        score = min_max_helper(successor, (depth + 1), pac_man_agent_index)
                        if score < current_score:
                            current_score = score
                    else:
                        score = min_max_helper(successor, depth, (agent_index + 1))
                        if score < current_score:
                            current_score = score
                return current_score

        if number_of_ghosts == 0:
            return possible_actions[pac_man_agent_index]
        else:
            for state in list_of_successors_pac_man:
                scores_for_actions.append(min_max_helper(state, 0, 1))
            for score in scores_for_actions:
                if score > max_score:
                    max_score = score
            iterator = 0
            for score in scores_for_actions:
                if score == max_score:
                    return possible_actions[iterator]
                else:
                    iterator = iterator + 1
        #################################################################################
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #################################################################################
        evaluation_function = self.evaluationFunction
        possible_actions = gameState.getLegalActions()
        number_of_agents = gameState.getNumAgents()
        number_of_ghosts = (number_of_agents - 1)
        list_of_successors_pac_man = []
        scores_for_actions = []
        pac_man_agent_index = 0
        search_agent_depth = self.depth
        max_score = -999999999999999
        alpha_value = -999999999999999
        beta_value = 999999999999999

        for action in possible_actions:
            list_of_successors_pac_man.append(gameState.generateSuccessor(pac_man_agent_index, action))

        # ==============================================================
        # Alpha Beta Helper Function
        # ==============================================================
        def alpha_beta(state, depth, agent_index, alpha_value, beta_value):
            # Create variables to store information
            legal_actions_for_agent = state.getLegalActions(agent_index)
            list_of_successors = []
            current_score = 0

            # If game is over or we are done with recursion
            if depth == search_agent_depth or state.isWin() or state.isLose():
                current_score = evaluation_function(state)
                return current_score

            # If agent is Maximizing player (Pac-Man)
            # ************************************************
            if agent_index == pac_man_agent_index:
                current_score = -99999999999999999999
                # Generate list of child nodes
                for action in legal_actions_for_agent:
                    successor_node = state.generateSuccessor(agent_index, action)
                    list_of_successors.append(successor_node)
                # Search child nodes
                for successor in list_of_successors:
                    if agent_index == number_of_ghosts:
                        score = alpha_beta(successor, (depth + 1), agent_index, alpha_value, beta_value)
                        if score > current_score:
                            current_score = score
                        if current_score > beta_value:
                            return current_score
                        if current_score > alpha_value:
                            alpha_value = current_score
                    else:
                        score = alpha_beta(successor, depth, (agent_index + 1), alpha_value, beta_value)
                        if score > current_score:
                            current_score = score
                        if current_score > beta_value:
                            return current_score
                        if current_score > alpha_value:
                            alpha_value = current_score
                return current_score

            # If agent is Minimizing player (Ghost)
            # ************************************************
            else:
                current_score = 99999999999999999999
                # Generate list of child nodes
                for action in legal_actions_for_agent:
                    successor_node = state.generateSuccessor(agent_index, action)
                    list_of_successors.append(successor_node)
                # Search child nodes
                for successor in list_of_successors:
                    if agent_index == number_of_ghosts:
                        score = alpha_beta(successor, (depth + 1), pac_man_agent_index, alpha_value, beta_value)
                        if score < current_score:
                            current_score = score
                        if current_score < alpha_value:
                            return current_score
                        if current_score < beta_value:
                            beta_value = current_score
                    else:
                        score = alpha_beta(successor, depth, (agent_index + 1), alpha_value, beta_value)
                        if score < current_score:
                            current_score = score
                        if current_score < alpha_value:
                            return current_score
                        if current_score < beta_value:
                            beta_value = current_score
                return current_score

        if number_of_ghosts == 0:
            return possible_actions[pac_man_agent_index]
        else:
            for state in list_of_successors_pac_man:
                scores_for_actions.append(alpha_beta(state, 0, 1, alpha_value, beta_value))
            for score in scores_for_actions:
                if score > max_score:
                    max_score = score
            iterator = 0
            for score in scores_for_actions:
                if score == max_score:
                    return possible_actions[iterator]
                else:
                    iterator = iterator + 1
        #################################################################################
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
