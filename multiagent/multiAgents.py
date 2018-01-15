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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        ############################################################################
        ## 0039026 ##
        #############
        # Important things to consider #
        #distance(currentPos - closestFood)
        #distance(newPos - closestFood)
        #distance(newPos - ghosts)
        #ghost.scared?

        #Variables
        score = 0
        closest_food = 99999
        closest_ghost = 99999
        current_food = currentGameState.getFood()

        for food in current_food.asList():
            temp_distance = util.manhattanDistance(food,newPos)
            if closest_food > temp_distance:
                closest_food = temp_distance

        score -= closest_food

        for ghost in newGhostStates:
            temp_distance = util.manhattanDistance(ghost.getPosition(), newPos)
            if closest_ghost>temp_distance:
                ghost_scaredTimer = ghost.scaredTimer
                if ghost_scaredTimer != 0:
                    if temp_distance <= 1:
                        score += 5555
                    else:
                        score += temp_distance
                else:
                    if temp_distance < 4:#danger!!
                        score += temp_distance*-10
        return score
        ############################################################################

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
        """
        "*** YOUR CODE HERE ***"
        ############################################################################
        ############################################################################
        ## 0039026 ##
        #############

        #############
        #Variables
        num_agents = gameState.getNumAgents()
        pacman_legalActions = gameState.getLegalActions(0)
        states = []
        pos_values = []
        max_value = 0


        #for action in pacman_legalActions:
        #    states.append(gameState.generateSuccessor(0, action))

        #for state in states:
         #   pos_values.append(self.min_function(0, 1, state))

        states = self.successor_states(0,gameState)
        pos_values = [self.min_function(0,1,state) for state in states]

        max_value = max(pos_values)
        index = pos_values.index(max_value)

        return pacman_legalActions[index]
    # ===================================

    # ===================================
    def min_function(self, current_depth, agent, state):

        #for action in agent_legalActions:
        #   states.append(state.generateSuccessors(agent, action))
        #--

        # --
        # if agent == num_agents - 1:
        #    for state in states:
        #       pos_values.append(self.max_function(current_depth + 1,0,state))
        #else:
        #   for state in states:
        #      pos_values.append(self.min_function(current_depth, agent + 1, state))
        # --
        #Variables
        agent_legalActions = state.getLegalActions(agent)
        num_agents = state.getNumAgents()
        pos_values = []

        if self.checkBaseCase(current_depth, state):
            return self.evaluationFunction(state)
        states = self.successor_states(agent,state)
        if agent >= num_agents - 1:
            pos_values = [self.max_function(current_depth + 1, 0, state) for state in states]
        else:
            pos_values = [self.min_function(current_depth, agent + 1, state) for state in states]

        return min(pos_values)
    # ===================================

    # ===================================
    def max_function(self,current_depth,agent, state):
        if self.checkBaseCase(current_depth,state):
            return self.evaluationFunction(state)

        pos_values = []
        states = self.successor_states(agent, state)
        pos_values =[self.min_function(current_depth,1,state) for state in states]

        return max(pos_values)
    # ===================================

    # ===================================
    def successor_states(self,agent,state):
        agent_legalActions = state.getLegalActions(agent)
        states=[state.generateSuccessor(agent, action) for action in agent_legalActions]
        return states
    # ===================================

    # ===================================
    def checkBaseCase(self, current_depth, state):
        if current_depth == self.depth or state.isLose() or state.isWin():
            return True
        else:
            return False
    # ===================================


############################################################################
############################################################################

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    ############################################################################
    ############################################################################
    ## 0039026 ##
    #############

    counter = 0
    # ===================================
    def getAction(self, gameState):
        ## Variables
        action = None

        ab_output = self.controller(gameState, 0, 0, -9999999, 9999999)
        action = ab_output[0]

        return action
    # ===================================

    # ===================================
    def max_function(self, agent, current_depth, state, alpha, beta):

            ## Variables
            t_action = None
            t_score = -999999
            t_succ = [t_action, t_score]

            if not state.getLegalActions(agent):
                return self.evaluationFunction(state)

            for action in state.getLegalActions(agent):
                m_state = state.generateSuccessor(agent, action)
                m_succ = self.controller(m_state, current_depth, agent + 1, alpha, beta)

                if type(m_succ) is float:
                    temp_score = m_succ
                else:
                    temp_score = m_succ[1]

                if temp_score > t_score or temp_score > beta:
                    t_score = temp_score
                    t_action = action
                    t_succ = [t_action, t_score]
                    if temp_score > beta:
                        return t_succ

                alpha = max(alpha, temp_score)
            return t_succ
    # ===================================

    # ===================================
    def min_function(self, agent, current_depth, state, alpha, beta):

            ##Variables
            t_action = None
            t_score = 999999
            t_succ = [t_action, t_score]

            if not state.getLegalActions(agent):
                return self.evaluationFunction(state)

            for action in state.getLegalActions(agent):
                m_state = state.generateSuccessor(agent, action)
                m_succ = self.controller(m_state, current_depth, agent + 1, alpha, beta)

                if type(m_succ) is float:
                    temp_score = m_succ
                else:
                    temp_score = m_succ[1]

                if temp_score < t_score or temp_score < alpha:
                    t_score = temp_score
                    t_action = action
                    t_succ = [t_action, t_score]
                    if temp_score < alpha:
                        return t_succ

                beta = min(beta, temp_score)
            return t_succ
    # ===================================

    # ===================================
    def controller(self,state, current_depth, agent, alpha, beta):

        ## Variables
        num_agents = state.getNumAgents()

        if agent == num_agents or agent == 0:
            if agent==num_agents:
                current_depth += 1
            if self.checkBaseCase(current_depth, state):
                g_output = self.evaluationFunction(state)
                return g_output
            g_output = self.max_function(0, current_depth, state, alpha, beta)
        else:
            if self.checkBaseCase(current_depth , state):
                g_output = self.evaluationFunction(state)
                return g_output
            g_output = self.min_function(agent, current_depth, state, alpha, beta)

        return g_output
    # ===================================

    # ===================================
    def successor_states(self, agent, state):
        agent_legalActions = state.getLegalActions(agent)
        states = [state.generateSuccessor(agent, action) for action in agent_legalActions]
        return states
    # ===================================

    # ===================================
    def checkBaseCase(self, current_depth, state):
        if (current_depth == self.depth or state.isWin()) or state.isLose():
            return True
        else:
            return False
    # ===================================
    ############################################################################
    ############################################################################


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    ############################################################################
    ############################################################################
    ## 0039026 ##
    #############

    counter = 0
    # ===================================
    def getAction(self, gameState):
        ## Variables
        action = None

        ab_output = self.controller(gameState, 0, 0)
        action = ab_output[0]

        return action
    # ===================================

    # ===================================
    def max_function(self, agent, current_depth, state):

            ## Variables
            t_action = None
            t_score = -999999
            t_succ = [t_action, t_score]

            if not state.getLegalActions(agent):
                return self.evaluationFunction(state)

            for action in state.getLegalActions(agent):
                m_state = state.generateSuccessor(agent, action)
                m_succ = self.controller(m_state, current_depth, agent + 1)

                if type(m_succ) is float:
                    temp_score = m_succ
                else:
                    temp_score = m_succ[1]

                if temp_score > t_score:
                    t_score = temp_score
                    t_action = action
                    t_succ = [t_action, t_score]
            return t_succ
    # ===================================

    # ===================================
    def exp_function(self, agent, current_depth, state):

            ##Variables
            t_action = None
            t_score = 0
            counter = 0
            t_succ = [t_action, t_score]

            if not state.getLegalActions(agent):
                return self.evaluationFunction(state)

            for action in state.getLegalActions(agent):
                m_state = state.generateSuccessor(agent, action)
                m_succ = self.controller(m_state, current_depth, agent + 1)

                if type(m_succ) is float:
                    temp_score = m_succ
                else:
                    temp_score = m_succ[1]
                t_score += temp_score
                counter += 1


            t_succ = [t_action, t_score/counter]
            return t_succ
    # ===================================

    # ===================================
    def controller(self,state, current_depth, agent):

        ## Variables
        num_agents = state.getNumAgents()

        if agent == num_agents or agent == 0:
            if agent==num_agents:
                current_depth += 1
            if self.checkBaseCase(current_depth, state):
                g_output = self.evaluationFunction(state)
                return g_output
            g_output = self.max_function(0, current_depth, state)
        else:
            if self.checkBaseCase(current_depth , state):
                g_output = self.evaluationFunction(state)
                return g_output
            g_output = self.exp_function(agent, current_depth, state)

        return g_output
    # ===================================

    # ===================================
    def successor_states(self, agent, state):
        agent_legalActions = state.getLegalActions(agent)
        states = [state.generateSuccessor(agent, action) for action in agent_legalActions]
        return states
    # ===================================

    # ===================================
    def checkBaseCase(self, current_depth, state):
        if (current_depth == self.depth or state.isWin()) or state.isLose():
            return True
        else:
            return False
    # ===================================
    ############################################################################
    ############################################################################

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    ############################################################################
    ## 0039026 ##
    #############
    # Important things to consider #
    # distance(currentPos - closestFood)
    # distance(newPos - closestFood)
    # distance(newPos - ghosts)
    # ghost.scared?

    # Variables
    score = 0
    scorem = 0
    closest_food = 99999
    closest_ghost = 99999
    current_food = currentGameState.getFood()
    pacman_position = list(currentGameState.getPacmanPosition())
    ghost_states = currentGameState.getGhostStates()

    for food in current_food.asList():
        temp_distance = util.manhattanDistance(food, pacman_position)
        if closest_food > temp_distance:
            closest_food = temp_distance

    score -= closest_food

    if(score == -99999):
        score = 0

    for ghost in ghost_states:
        temp_distance = util.manhattanDistance(ghost.getPosition(), pacman_position)
        if closest_ghost > temp_distance:
            ghost_scaredTimer = ghost.scaredTimer
            if ghost_scaredTimer != 0:
                if temp_distance <= 1:
                    score += 5
                else:
                    score += temp_distance
            else:
                if temp_distance < 4:  # danger!!
                    score += temp_distance * -3

    return score + currentGameState.getScore()
############################################################################

# Abbreviation
better = betterEvaluationFunction

