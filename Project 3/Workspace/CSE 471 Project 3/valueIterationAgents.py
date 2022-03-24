# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        ################################################################################################################
        # Create a variable to store the MDP
        markov_decision_process = self.mdp

        # Create a variable to store the number of iterations to perform value iteration
        iterations_to_perform = self.iterations

        # Create a variable to store a list of the states in the MDP
        states = markov_decision_process.getStates()

        # ==============================================================================================================
        #   For each iteration of value iteration, compute best Q-Value for each state and update the value function
        # ==============================================================================================================
        while iterations_to_perform is not 0:  # While there are still iterations of value iteration left to perform...

            # Create a dictionary to store best Q-Values for each state during the iteration
            value_function = util.Counter()  # Counter initializes a dictionary with default values of 0

            # **********************************************************************************************************
            #   For each state in the MDP, compute Q-Values for each action and store the best in the value function
            # **********************************************************************************************************
            for state in states:

                # Create a variable to store the best Q-Value for the state
                best_q_value = None

                # ------------------------------------------------------------------------------------------------------
                #   If the state is a terminal state, set the best Q-Value to 0
                # ------------------------------------------------------------------------------------------------------
                if markov_decision_process.isTerminal(state):
                    best_q_value = 0
                # ------------------------------------------------------------------------------------------------------
                #   If the state is NOT a terminal state, compute the Q-Value for each action and store the best value
                # ------------------------------------------------------------------------------------------------------
                else:  # Else, the state is NOT a terminal state

                    # Create a variable to store a list of possible actions for the state
                    possible_actions = markov_decision_process.getPossibleActions(state)

                    # **************************************************************************************************
                    #   For each action possible from the state, compute the Q-Value and store the best value
                    # **************************************************************************************************
                    for action in possible_actions:

                        # Compute Q-Value of the action in the state and store it
                        q_value = self.getQValue(state, action)

                        # ----------------------------------------------------------------------------------------------
                        #   If the best Q-Value has NOT been decided yet, set it to the calculated Q-Value
                        # ----------------------------------------------------------------------------------------------
                        if best_q_value is None:
                            best_q_value = q_value
                        # ----------------------------------------------------------------------------------------------
                        #   If the best Q-Value has been decided, check if it is less than the calculated Q-Value
                        # ----------------------------------------------------------------------------------------------
                        else:
                            # If the best Q-Value is less than the calculated Q-Value, set best as calculated Q-Value
                            if best_q_value < q_value:
                                best_q_value = q_value

                # Store the best Q-Value for the state in the value function dictionary
                value_function[state] = best_q_value

            # Update the values of each state in the MDP
            self.values = value_function

            # Reduce the number of iterations to perform after completing an iteration
            iterations_to_perform = (iterations_to_perform - 1)
        ################################################################################################################


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        ################################################################################################################
        # Create a variable to store the current state and given action
        current_state = state
        given_action = action

        # Create a variable to store the MDP
        markov_decision_process = self.mdp

        # Create a variable to store the discount factor
        discount_factor = self.discount

        # Create a variable to store the value function
        value_function = self.values

        # Create a variable to store a list of pairs containing a reachable state and the transition probability
        transition_function = markov_decision_process.getTransitionStatesAndProbs(current_state, given_action)

        # Create a variable to store a list of calculated reward values for each transition state
        calculated_reward_values = []

        # Create a variable to store the Q-Value
        q_value = 0

        # ==============================================================================================================
        #   For each state reachable from the current state after taking the given action,
        # ==============================================================================================================
        for reachable_state in transition_function:

            # Store the data of the reachable state (transition state)
            transition_state = reachable_state[0]

            # Store the probability of reaching the transition state with the given action
            transition_probability = reachable_state[1]

            # Store the immediate reward of reaching the transition state from the current state with the given action
            immediate_reward = markov_decision_process.getReward(current_state, given_action, transition_state)

            # Store the future reward value of the transition state (if acting optimally) given by the value function
            future_reward = value_function[transition_state]

            # Compute the discounted future reward and store it
            discounted_future_reward = (discount_factor * future_reward)

            # Calculate the reward value for the transition state
            reward_value = (transition_probability * (immediate_reward + discounted_future_reward))

            # Append the reward value to the list of calculated reward values for each transition state
            calculated_reward_values.append(reward_value)

        # Calculate the summation of the rewards for each transition state
        for value in calculated_reward_values:
            q_value = q_value + value

        # Return the Q-Value
        return q_value
        ################################################################################################################
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        ################################################################################################################
        # Create a variable to store the current state
        current_state = state

        # Create a variable to store the MDP
        markov_decision_process = self.mdp

        # If the current state is a terminal state, return None (no legal actions)
        if markov_decision_process.isTerminal(current_state):
            return None

        # Create a variable to store the list of possible actions from the current state
        possible_actions = markov_decision_process.getPossibleActions(current_state)

        # Create a variable to store the best Q-Value
        best_q_value = None

        # Create a variable to store the best action
        best_action = None

        # ==============================================================================================================
        #   For each action possible from current state...
        # ==============================================================================================================
        for action in possible_actions:

            # Use the value function to calculate the Q-Value for the possible action in the current state
            q_value = self.computeQValueFromValues(current_state, action)

            # ----------------------------------------------------------------------------------------------------------
            #   If the best Q-Value has NOT been decided yet, set it to the calculated Q-Value
            # ----------------------------------------------------------------------------------------------------------
            if best_q_value is None:
                best_q_value = q_value
            # ----------------------------------------------------------------------------------------------------------
            #   If the best Q-Value has been decided, check if it is less than the calculated Q-Value
            # ----------------------------------------------------------------------------------------------------------
            else:
                # If the best Q-Value is less than the calculated Q-Value, set best Q-Value to the calculated Q-Value
                if best_q_value < q_value:
                    best_q_value = q_value

            # If the calculated Q-Value is the best Q-Value, set the possible action to the best action
            if q_value is best_q_value:
                best_action = action

        # Return the best action
        return best_action
        ################################################################################################################
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        ################################################################################################################
        # Create a variable to store the MDP
        markov_decision_process = self.mdp

        # Create a variable to store the number of iterations to perform asynchronous value iteration
        iterations_to_perform = self.iterations

        # Create a variable to store a list of the states in the MDP
        states = markov_decision_process.getStates()

        # Create a variable to store the number of states in the MDP
        number_of_states = len(states)

        # Create a variable to store the index of a state in the MDP
        state_index = 0

        # ==============================================================================================================
        #   For each iteration of asynchronous value iteration, compute best Q-Value for a state & update state values
        # ==============================================================================================================
        while iterations_to_perform is not 0:  # While there are still iterations left to perform...

            # Create a variable to store the state we want to update during the current iteration
            state = states[state_index]

            # Create a variable to store the best Q-Value for the state
            best_q_value = None

            # ----------------------------------------------------------------------------------------------------------
            #   If the state is a terminal state, set the best Q-Value to 0
            # ----------------------------------------------------------------------------------------------------------
            if markov_decision_process.isTerminal(state):
                best_q_value = 0
            # ----------------------------------------------------------------------------------------------------------
            #   If the state is NOT a terminal state, compute the Q-Value for each action and store the best value
            # ----------------------------------------------------------------------------------------------------------
            else:  # Else, the state is NOT a terminal state

                # Create a variable to store a list of possible actions for the state
                possible_actions = markov_decision_process.getPossibleActions(state)

                # ******************************************************************************************************
                #   For each action possible from the state, compute the Q-Value and store the best value
                # ******************************************************************************************************
                for action in possible_actions:

                    # Compute Q-Value of the action in the state and store it
                    q_value = self.getQValue(state, action)

                    # --------------------------------------------------------------------------------------------------
                    #   If the best Q-Value has NOT been decided yet, set it to the calculated Q-Value
                    # --------------------------------------------------------------------------------------------------
                    if best_q_value is None:
                        best_q_value = q_value
                    # --------------------------------------------------------------------------------------------------
                    #   If the best Q-Value has been decided, check if it is less than the calculated Q-Value
                    # --------------------------------------------------------------------------------------------------
                    else:
                        # If the best Q-Value is less than the calculated Q-Value, set best as calculated Q-Value
                        if best_q_value < q_value:
                            best_q_value = q_value

            # Update the value of the state in the MDP
            self.values[state] = best_q_value

            # Reduce the number of iterations to perform after completing an iteration
            iterations_to_perform = (iterations_to_perform - 1)

            # Increment the state index in order to update the next state during the next iteration
            state_index = (state_index + 1)

            # If the state index has already reached the last state, set index back to the first state
            if state_index == number_of_states:
                state_index = 0
        ################################################################################################################

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        ################################################################################################################
        states = self.mdp.getStates()
        p_dict = util.Counter()

        # Init dict for each state with set
        for state in states:
            p_dict[state] = set()

        # for each state
        for state in states:
            # Get actions
            actions = self.mdp.getPossibleActions(state)
            # for each action
            for action in actions:
                # Get transition states
                transition_states = self.mdp.getTransitionStatesAndProbs(state, action)
                # for each transition state
                for transition_state in transition_states:
                    t_state = transition_state[0]
                    predecessors = p_dict[t_state]
                    if transition_state[1] is not 0:
                        predecessors.add(state)
                    p_dict[t_state] = predecessors

        pq = util.PriorityQueue()

        for s in states:
            if not self.mdp.isTerminal(s):
                current_value = self.values[s]
                best_q_value = None
                s_actions = self.mdp.getPossibleActions(s)
                for action in s_actions:
                    q_value = self.computeQValueFromValues(s, action)

                    if best_q_value is None:
                        best_q_value = q_value
                    else:
                        if best_q_value < q_value:
                            best_q_value = q_value

                diff = current_value - best_q_value
                if diff < 0:
                    diff = diff*(-1)

                pq.push(s, (diff*1))

        for i in range(0, self.iterations):
            if pq.isEmpty() is False:
                s = pq.pop()
                if self.mdp.isTerminal(s) is False:
                    best_q_value = None
                    s_actions = self.mdp.getPossibleActions(s)
                    for action in s_actions:
                        q_value = self.computeQValueFromValues(s, action)

                        if best_q_value is None:
                            best_q_value = q_value
                        else:
                            if best_q_value < q_value:
                                best_q_value = q_value
                    self.values[s] = best_q_value

                for p in p_dict[s]:
                    current_value = self.values[p]
                    best_q_value = None
                    p_actions = self.mdp.getPossibleActions(p)
                    for action in p_actions:
                        q_value = self.computeQValueFromValues(p, action)

                        if best_q_value is None:
                            best_q_value = q_value
                        else:
                            if best_q_value < q_value:
                                best_q_value = q_value
                    diff = current_value - best_q_value
                    if diff < 0:
                        diff = diff * (-1)
                    if diff > self.theta:
                        pq.push(p, (diff*-1))
        ################################################################################################################
