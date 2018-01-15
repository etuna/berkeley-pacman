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

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #####################################################
        # 0039026 #
        ###########

        # MDP ops---------
        #mdp.getStates()
        #mdp.getPossibleActions(state)
        #mdp.getTransitionStatesAndProbs(state, action)
        #mdp.getReward(state, action, nextState)
        #mdp.isTerminal(state)


        # print "get states -------"
        # mdp.getStates()
        # print "get pos actions -------"
        # mdp.getPossibleActions(state)
        # print "get transition states and probs -------"
        # mdp.getTransitionStatesAndProbs(state, action)
        # print "get reward -------"
        # mdp.getReward(state, action, nextState)
        # print "get isTerminal -------"
        # mdp.isTerminal(state)

        self.init_values(mdp)


        #####################################################

    def init_values(self,mdp):

        for i in range(self.iterations):

            values = util.Counter()
            states = mdp.getStates()

            for state in states:

                max_value = None
                value = None
                maction = None

                if mdp.isTerminal(state):
                    # print "end"
                    #self.values = values
                    None
                else:
                    actions = mdp.getPossibleActions(state)
                    max_value = self.getValueOrAction(actions,state,0)
                    values[state] = max_value

            self.values = values



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
        #####################################################
        # 0039026 #
        ###########

        Q = 0
        gamma = self.discount
        values = self.values
        states_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        #print states_probs

        for next_state, prob in states_probs:

            # Qk+1 = E prob * (reward + gamma* [(max a)] Qk)
            # reward = s, a, s'
            Q += prob * (self.mdp.getReward(state, action, next_state) + (gamma * values[next_state]))
            #print V

        return Q
        ####################################################

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #####################################################
        # 0039026 #
        ###########

        actions = self.mdp.getPossibleActions(state)
        """maction = None
        max_value = 0

        for action in actions:
            value = self.computeQValueFromValues(state, action)

            if max_value == 0 or max_value<value:
                max_value = value
                maction = action

        return maction"""

        return self.getValueOrAction(actions,state,1)


        #####################################################

    def getValueOrAction(self,actions,state,op):
        # 0 max_val , 1 action
        maction = None
        max_value = 0
        for action in actions:
            value = self.computeQValueFromValues(state, action)

            if max_value == 0 or max_value<value:
                max_value = value
                maction = action

        if op == 0:
           return max_value
        else:
            return maction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
