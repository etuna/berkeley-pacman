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
        ################################################
        # 0039026 #
        ###########

        # epsilon :self.epsilon
        # alpha : self.alpha
        # gamma : self.discount

        self.Q_values = util.Counter()
        ################################################

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        ################################################
        # 0039026 #
        ###########

        if (state,action) not in self.Q_values:
            self.Q_values[(state,action)] = 0.0

        return self.Q_values[(state, action)]
        ################################################


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        ################################################
        # 0039026 #
        ###########
        """
        actions = self.getLegalActions(state)
        max_value = -9999999

        if len(actions) == 0:
            return 0

        for action in actions:
            Q_value = self.getQValue(state, action)
            #print Q_value
            #print "q-------max"
            #print max_value

            if max_value==-99999999 or Q_value > max_value:
                max_value = Q_value

            #print max_value
            #print "second max val"

        return max_value
        """
        # op = 0 is value
        value = self.getValueOrAction(state, 0)
        return value

        ################################################

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        ################################################
        # 0039026 #
        ###########
        """
        actions = self.getLegalActions(state)
        max_value = -9999999
        maction = None

        if len(actions) == 0:
            return 0

        for action in actions:
            Q_value = self.getQValue(state, action)
            # print Q_value
            # print "q-------max"
            # print max_value

            if max_value == -99999999 or Q_value > max_value:
                max_value = Q_value
                maction = action

                # print max_value
                # print "after if, max val----"
        return maction
        """
        # op = 1 is action
        action = self.getValueOrAction(state,1)
        return action


        ################################################

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
        ################################################
        # 0039026 #
        ###########
        """Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)"""

        #util.flipCoin(p)
        #print util.flipCoin(10)
        #print self.epsilon
        bool = util.flipCoin(self.epsilon)

        if not bool:
            #print "COINhere false ----------------"
            action = self.computeActionFromQValues(state)
        else:
            #print "COINhere true ----------------"
            action = random.choice(legalActions)

        return action

        ################################################


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        ################################################
        # 0039026 #
        ###########

        """
        Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
        """
        alpha = self.alpha
        epsilon = self.epsilon
        gamma = self.discount

        Q = self.getQValue(state, action)
        Q_value = self.getValueOrAction(nextState,0)
        #print Q
        #print Q_value


        #sample = R(s,a,s') + (gamma * max (a') (Q(s',a')))
        sample = reward + (gamma*Q_value)

        #Q(s, a) = ((1 - alpha) * Q(s, a)) + (alpha * sample)
        self.Q_values[(state, action)] = ((1 - alpha) * Q) + (alpha * sample)


        ################################################

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


    def getValueOrAction(self,state,op):
        actions = self.getLegalActions(state)
        max_value = -9999999
        maction = None

        if len(actions) == 0:
            if op==1:
                return None
            return 0

        for action in actions:
            Q_value = self.getQValue(state, action)
            # print Q_value
            # print "q-------max"
            # print max_value

            if max_value == -99999999 or Q_value > max_value:
                max_value = Q_value
                maction = action

                # print max_value
                # print "after if, max val----"

        # 0 value , 1 action
        if op == 0:
            return max_value
        else:
            return maction


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
        ################################################
        # 0039026 #
        ###########
        #class IdentityExtractor(FeatureExtractor):
        #    def getFeatures(self, state, action):

        Q = 0
        #print self.weights

        features = self.featExtractor.getFeatures(state, action)
        # print features

        #Q(state, action) = w * featureVector
        for f in features:
            Q += self.weights[f] * features[f]

        return Q
        ################################################

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        ################################################
        # 0039026 #
        ###########


        alpha = self.alpha
        epsilon = self.epsilon
        gamma = self.discount
        weights = self.weights

        #Wm = Wm + alpha*[reward + gamma*(maxa)Q(s',a') - Q(s,a] * fm(s,a)

        features = self.featExtractor.getFeatures(state,action)

        #Q-Values at iteration 3 for action 'exit' are NOT correct. ??
        #if action == "exit":
        #    print self.getQValue(state,action)

        #print features

        for k in features.keys():
            #print k
            #weights[k] += weights[k] + alpha*(reward + gamma * self.getValueOrAction(nextState,0) - self.getQValue(state,action))*features[k]
            features[k] = alpha*(reward + gamma * self.getValueOrAction(nextState,0) - self.getQValue(state,action))*features[k]


        for f in features:
            weights[f] += features[f]

        self.weights = weights

        #self.weights = weights
        ################################################

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
        ################################################
        # 0039026 #
        ###########

            #print self.weights


        ################################################
            pass
