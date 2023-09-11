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

def depthFirstSearch(problem: SearchProblem):
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
    edges = util.Stack()
    exploredEdges = set()
    edges.push((problem.getStartState(), [])) # push the initial state with no actions (origin)
    while not edges.isEmpty():
        currentEdge, actions = edges.pop() # get the current edge and actions to origin
        if problem.isGoalState(currentEdge):
           return actions
           
         # if not goal state
        exploredEdges.add(currentEdge)

        for successorEdge, action, stepCost in problem.getSuccessors(currentEdge):
           if successorEdge not in exploredEdges and (successorEdge, action) not in edges.list:
               newActions = actions.copy()
               newActions.append(action)
               edges.push((successorEdge, newActions))

    # no solution 
    return []

def breadthFirstSearch(problem: SearchProblem):
    edges = util.Queue() 
    exploredEdges = set()
    
    edges.push((problem.getStartState(), []))  # push the initial state with no actions (origin)
    
    while not edges.isEmpty():
        currentEdge, actions = edges.pop()  # get the current edge and actions to origin
        
        if currentEdge in exploredEdges:  # Skip if this state has already been explored
            continue
        
        if problem.isGoalState(currentEdge):
            return actions
           
        # if not goal state
        exploredEdges.add(currentEdge)

        for successorEdge, action, _ in problem.getSuccessors(currentEdge):
            if successorEdge not in exploredEdges:
                newActions = actions.copy()
                newActions.append(action)
                edges.push((successorEdge, newActions))

    # no solution 
    return []


def uniformCostSearch(problem: SearchProblem):
    edges = util.PriorityQueue()  # Use a PriorityQueue for assigning step cost
    exploredEdges = set()
    
    edges.push((problem.getStartState(), []), 0)  # push the initial state with no actions and cost 0
    
    while not edges.isEmpty():
        currentEdge, actions = edges.pop()  # get the current edge and actions to origin
        
        if currentEdge in exploredEdges:  # Skip if this state has already been explored
            continue
        
        if problem.isGoalState(currentEdge):
            return actions
           
        # if not goal state
        exploredEdges.add(currentEdge)

        for successorEdge, action, _ in problem.getSuccessors(currentEdge):
            if successorEdge not in exploredEdges:
                newActions = actions.copy()
                newActions.append(action)
                newCost = problem.getCostOfActions(newActions)  # Get the total cost of the new actions
                edges.push((successorEdge, newActions), newCost)  # Push with the new cost

    # no solution 
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def manhattanHeuristic(state, problem=None):

    return 0



def aStarSearch(problem, heuristic=manhattanHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    edges = util.PriorityQueue()
    exploredEdges = set()

    startState = problem.getStartState()
    edges.push((startState, []), heuristic(startState, problem))

    while not edges.isEmpty():
        currentState, actions = edges.pop()

        if problem.isGoalState(currentState):
            return actions

        if currentState not in exploredEdges:
            exploredEdges.add(currentState)

            for successor, action, stepCost in problem.getSuccessors(currentState):
                newActions = actions + [action]
                g = problem.getCostOfActions(newActions)
                h = heuristic(successor, problem)
                f = g + h

                edges.push((successor, newActions), f)

    return []



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
