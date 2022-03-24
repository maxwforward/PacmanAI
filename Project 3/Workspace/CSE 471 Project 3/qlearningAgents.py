# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        ################################################################################################################
        # Initialize Q-Values
        self.q_values = util.Counter()  # Counter initializes a dictionary with default values of 0
        ################################################################################################################

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        ################################################################################################################
        # Create a variable to store the Q-Function which gives the Q-Value of a state/action pair
        q_function = self.q_values

        # Create a variable to store a pair consisting of the state and action
        state_action_pair = (state, action)

        # If we have NEVER seen the state, return 0.0
        if state_action_pair not in q_function:
            return 0.0

        # Get the Q-Value for the state/action pair from the Q-Function
        q_value = q_function[state_action_pair]

        # Return the Q-Value
        return q_value
        ################################################################################################################
        # util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        ################################################################################################################
        # Create a variable to store a list of legal actions for the state
        legal_actions = self.getLegalActions(state)
        number_of_legal_actions = len(legal_actions)
        if number_of_legal_actions is 0:
            return 0.0

        max_value = None

        for action in legal_actions:
            value = self.getQValue(state, action)
            if max_value is None:
                max_value = value
            else:
                if max_value < value:
                    max_value = value

        return max_value
        ################################################################################################################
        # util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        ################################################################################################################
        legal_actions = self.getLegalActions(state)
        number_of_legal_actions = len(legal_actions)
        if number_of_legal_actions is 0:
            return None

        max_value = None
        best_action = None

        for action in legal_actions:
            value = self.getQValue(state, action)
            if max_value is None:
                max_value = value
                best_action = action
            else:
                if max_value < value:
                    max_value = value
                    best_action = action

        return best_action
        ################################################################################################################
        # util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        ################################################################################################################
        if len(legalActions) is not 0:
            epsilon = self.epsilon
            result = util.flipCoin(epsilon)

            if result is True:
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)
        ################################################################################################################
        # util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        ################################################################################################################
        s_a = (state, action)
        q_s_a = self.getQValue(state, action)
        alpha = self.alpha
        r_s = reward
        discount_factor = self.discount
        s_prime = nextState
        q_s_prime_a_prime_list = []
        a_primes = self.getLegalActions(s_prime)

        for a_prime in a_primes:
            q_s_prime_a_prime = self.getQValue(s_prime, a_prime)
            q_s_prime_a_prime_list.append(q_s_prime_a_prime)

        if len(q_s_prime_a_prime_list) is 0:
            max_q_s_prime_a_prime = 0
        else:
            max_q_s_prime_a_prime = max(q_s_prime_a_prime_list)

        updated_q_value = q_s_a + alpha * (r_s + discount_factor * max_q_s_prime_a_prime - q_s_a)
        self.q_values[s_a] = updated_q_value
        ################################################################################################################
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        ################################################################################################################
        ################################################################################################################
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        ################################################################################################################
        ################################################################################################################
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            ############################################################################################################
            ############################################################################################################
            pass
