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

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #################################################################################
    # 0039226 #
    ###########
    """Variables"""
    visited = []
    state = problem.getStartState()
    frontier = util.Stack()
    start_node = dict()
    start_node["parent"] = None
    start_node["action"] = None
    start_node["state"] = state
    in_visited = 0
    actions = []
    """start_node = {'state':state,'action':None, 'parent':None}"""
    """----------------------------"""
    frontier.push(start_node)
    node = dict()

    while not frontier.isEmpty():
        node = frontier.pop()
        state = node["state"]
        for nodem in visited:
            if nodem["state"] == state:
                in_visited = 1
        if in_visited == 0:
            visited.insert(0, node)
        else:
            in_visited = 0
            continue

        if problem.isGoalState(state):
            break

        for child in problem.getSuccessors(state):
            for noder in visited:
                if noder["state"] == child[0]:
                    break
            child_node = {'state': child[0], 'action': child[1], 'parent': node}
            frontier.push(child_node)
    while not node["action"] == None:
        actions.insert(0, node["action"])
        node = node["parent"]

    return actions
    #################################################################################



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #################################################################################
    # 0039226 #
    ###########
    """Variables"""
    visited = []
    state = problem.getStartState()
    frontier = util.Queue()
    start_node = dict()
    start_node["parent"] = None
    start_node["action"] = None
    start_node["state"] = state
    in_visited = 0
    """start_node = {'state':state,'action':None, 'parent':None}"""
    """----------------------------"""
    node = dict()
    frontier.push(start_node)
    while not frontier.isEmpty():
        start_node = frontier.pop()
        state = start_node["state"]

        for nodem in visited:
            if nodem["state"] == state:
                in_visited = 1

        if in_visited == 0:
            visited.insert(0, start_node)
        else :
            in_visited = 0
            continue

        if problem.isGoalState(state):
            break

        for child in problem.getSuccessors(state):
            for noder in visited:
                if noder["state"] == child[0]:
                    break
            child_node={'state': child[0], 'action': child[1], 'parent': start_node}
            frontier.push(child_node)

    actions=[]
    while not start_node["action"] == None:
        actions.insert(0, start_node["action"])
        start_node = start_node["parent"]

    return actions
    #################################################################################



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #################################################################################
    # 0039226 #
    ###########
    """""""""Variables"""""""""
    start_node = {'state':problem.getStartState(), 'action':None, 'parent':None , 'cost':0}
    state = start_node["state"]
    visited = dict()
    nodem = dict()
    child_node = dict()
    actions = []
    frontier = util.PriorityQueue()

    frontier.push(start_node, start_node["cost"])

    while not frontier.isEmpty():
        node = frontier.pop()
        state = node["state"]
        cost = node["cost"]
        if visited.has_key(hash(state)):
            continue

        visited[hash(state)]=True
        if problem.isGoalState(state):
            break

        for child in problem.getSuccessors(state):
            if not visited.has_key(hash(child[0])):
                child_node = {'state':child[0], 'action':child[1], 'parent':node, 'cost':child[2]+cost}
                frontier.push(child_node,child_node["cost"])

    while node["action"] != None:
        actions.insert(0,node["action"])
        node = node["parent"]
    return actions
    #################################################################################




def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #################################################################################
    # 0039226 #
    ###########
    ##Variables
    visited = dict()
    state = problem.getStartState()
    """(f=g+h) , priority:f"""
    frontier = util.PriorityQueue()
    node = dict()
    actions = []
    ##---
    start_node = {'state':state, 'action':None, 'cost':0, 'parent':None, 'heuristic':heuristic(state, problem)}
    frontier.push(start_node,start_node["cost"]+ start_node["heuristic"])

    while not frontier.isEmpty():
        node = frontier.pop()
        cumulative_cost = node["cost"]
        _heuristic = node["heuristic"]
        state = node["state"]

        if visited.has_key(hash(state)):
            continue
        else:
            visited[hash(state)] = True
            if problem.isGoalState(state):
                break

            for child in problem.getSuccessors(state):
                if not visited.has_key(hash(child[0])):
                    child_node = {'state': child[0], 'action': child[1], 'cost':(child[2]+cumulative_cost) , 'parent':node ,  'heuristic':heuristic(child[0], problem)}
                    frontier.push(child_node, child_node["cost"]+ child_node["heuristic"])

    while node["action"] != None:
        actions.insert(0, node["action"])
        node = node["parent"]
    return actions
    #################################################################################





# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
