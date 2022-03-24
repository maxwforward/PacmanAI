# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    ####################################################################################################################
    # When discount is closer to 1, the agent will prioritize optimal future rewards rather than closer rewards
    answerDiscount = 0.9  # Unchanged
    # When noise is 0, the agent will not end up in unintended states after performing actions and therefore take risks
    answerNoise = 0  # Changed from 0.2 to 0
    ####################################################################################################################
    return answerDiscount, answerNoise

def question3a():  # Prefer the close exit (+1), risking the cliff (-10)
    ####################################################################################################################
    # When discount is closer to 0, the agent will prioritize the closer rewards
    answerDiscount = 0.1  # Changed from "None" to 0.1
    # When noise is 0, the agent will not end up in unintended states after performing actions and therefore take risks
    answerNoise = 0  # Changed from "None" to 0
    # When living reward is negative, the agent will optimize a policy with less actions
    answerLivingReward = -0.1  # Changed from "None" to -0.1
    ####################################################################################################################
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():  # Prefer the close exit (+1), avoiding the cliff (-10)
    ####################################################################################################################
    # When discount is closer to 0, the agent will prioritize the closer rewards
    answerDiscount = 0.2  # Changed from "None" to 0.2
    # When noise is not 0, the agent may end up in unintended states after performing actions and therefore avoid risks
    answerNoise = 0.2  # Changed from "None" to 0.2
    # When living reward is positive, the agent will optimize a policy with more actions
    answerLivingReward = 0.1  # Changed from "None" to 0.1
    ####################################################################################################################
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():  # Prefer the distant exit (+10), risking the cliff (-10)
    ####################################################################################################################
    # When discount is closer to 1, the agent will prioritize optimal future rewards rather than closer rewards
    answerDiscount = 0.9  # Changed from "None" to 0.9
    # When noise is 0, the agent will not end up in unintended states after performing actions and therefore take risks
    answerNoise = 0  # Changed from "None" to 0
    # When living reward is negative, the agent will optimize a policy with less actions
    answerLivingReward = -0.1  # Changed from "None" to -0.1
    ####################################################################################################################
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():  # Prefer the distant exit (+10), avoiding the cliff (-10)
    ####################################################################################################################
    # When discount is closer to 1, the agent will prioritize optimal future rewards rather than closer rewards
    answerDiscount = 0.9  # Changed from "None" to 0.9
    # When noise is not 0, the agent may end up in unintended states after performing actions and therefore avoid risks
    answerNoise = 0.2  # Changed from "None" to 0.2
    # When living reward is positive, the agent will optimize a policy with more actions
    answerLivingReward = 0.1  # Changed from "None" to 0.1
    ####################################################################################################################
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():  # Avoid both exits and the cliff (so an episode should never terminate)
    ####################################################################################################################
    # When discount is 0, the agent will only prioritize actions that result in the highest immediate reward
    answerDiscount = 0  # Changed from "None" to 0
    # When noise is 0, the agent will not end up in unintended states after performing actions and therefore take risks
    answerNoise = 0  # Changed from "None" to 0
    # When living reward is greater than the rewards of exit states, the agent will optimize a policy that does not exit
    answerLivingReward = 20  # Changed from "None" to 20
    ####################################################################################################################
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    ####################################################################################################################
    answerEpsilon = None
    answerLearningRate = None
    ####################################################################################################################
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
