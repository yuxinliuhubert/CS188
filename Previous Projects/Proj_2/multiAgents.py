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
import math
from math import inf

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        score = successorGameState.getScore()
        # closest food
        all_manhattanDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if len(all_manhattanDistances) != 0:
            min_Manhattan_Distances = min(all_manhattanDistances)
                        # Set up greed (attraction to food, higher score if closer to food, reciprocal)
            if min_Manhattan_Distances != 0:
                score += 1.0/min_Manhattan_Distances
        
        
        # closest ghost
        newManhattanDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates] or [inf]
        if len(newManhattanDistances) != 0:
            closest_ghost_distance = min(newManhattanDistances)
        # get the scare time of the closest ghost
            closest_ghost_index = newManhattanDistances.index(closest_ghost_distance)
            closest_ghost_scared_time = newScaredTimes[closest_ghost_index]

     

            # penalty starts
            # penalize heavily if ghost is close
            if closest_ghost_scared_time == 0 and closest_ghost_distance <= 1:
                score -= 1000

            # if the ghost is scared and close, then pacman should go towards it
            if closest_ghost_scared_time > 0 and closest_ghost_distance <= closest_ghost_scared_time:
                score += 300

                




        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
               # Recursive function for Minimax algorithm
        def minimax(agent, depth, gameState):
            # Base case: if the game is over or depth is zero, return the evaluated score
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            # For Pacman (maximizing agent)
            if agent == 0:
                return max(minimax(1, depth, gameState.generateSuccessor(agent, new_action))
                           for new_action in gameState.getLegalActions(agent))
            
            # For ghosts (minimizing agents)
            else:
                next_agent = agent + 1  # Move to the next agent
                if agent == gameState.getNumAgents() - 1:  # If this is the last ghost, move to Pacman and reduce the depth
                    next_agent = 0
                    depth -= 1
                return min(minimax(next_agent, depth, gameState.generateSuccessor(agent, new_action))
                           for new_action in gameState.getLegalActions(agent))

        # Start the Minimax algorithm for the current gameState and return the best action
        best_score = -inf
        best_action = None
        for action in gameState.getLegalActions(0):  # 0 index for Pacman
            score = minimax(1, self.depth, gameState.generateSuccessor(0, action))
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # maximazing agent
        def max_value(gameState, depth, alpha, beta):
            # end the game early if winning or losing is imminent
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            
            v = float("-inf")
            for action in gameState.getLegalActions(0):
                v = max(v, min_value(gameState.generateSuccessor(0, action), 1, depth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        # minimizing agent
        def min_value(gameState, agentIndex, depth, alpha, beta):
            # end the game early if winning or losing is imminent
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            
            v = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    v = min(v, max_value(gameState.generateSuccessor(agentIndex, action), depth - 1, alpha, beta))

                else:
                    v = min(v, min_value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
        
        alpha = float("-inf")
        beta = float("inf")
        bestValue = float("-inf")
        bestAction = Directions.STOP

        # best action for pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = min_value(successor, 1, self.depth, alpha, beta)

            # maximizing the value
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
         # maximazing agent
        def max_value(gameState, depth):
            # end the game early if winning or losing is imminent
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            
            v = float("-inf")
            for action in gameState.getLegalActions(0):
                # cannot skip anymore
                v = max(v, exp_value(gameState.generateSuccessor(0, action), 1, depth))
            return v
        
        # chance agent
        def exp_value(gameState, agentIndex, depth):
            # end the game early if winning or losing is imminent
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            numActions = len(gameState.getLegalActions(agentIndex))
            v = 0
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    v += max_value(gameState.generateSuccessor(agentIndex, action), depth - 1)/numActions
                else:
                    v += exp_value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)/numActions
        
            return v
        
        bestValue = float("-inf")
        bestAction = Directions.STOP

        # best action for pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = exp_value(successor, 1, self.depth)

            # maximizing the value
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation function gets the current position of pacman, the food, the ghost, and the scared time of the ghosts. Then it initializes a score from
    the current game state. The score is then updated based on the distance to the nearest food, the distance to the ghosts, and the number of remaining food pellets. 
    We penalize the score heavily if the ghost is too close to pacman, and we reward somewhat generously if the ghost is scared and close to pacman. In addition, the inverse of 
    the nearst food distance is added as an incentive for the pacman to move towards it. The final score after all the updates will be a heuristic for the pacman to choose the best action.
    """
    "*** YOUR CODE HERE ***"
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    curGhostStates = currentGameState.getGhostStates()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]
    
    # Initialize score
    score = currentGameState.getScore()
    
    # Distance to nearest food
    foodList = curFood.asList()
    if len(foodList) > 0:
        nearestFoodDist = min([manhattanDistance(curPos, food) for food in foodList])
        # The closer the food, the higher the score
        score += 1.0 / nearestFoodDist
    
    # Distance to ghosts
    ghostDistances = [manhattanDistance(curPos, ghost.getPosition()) for ghost in curGhostStates]
    # enumerate: TypeError: cannot unpack non-iterable float object if without
    for i, dist in enumerate(ghostDistances):
        # If ghosts are not scared
        if curScaredTimes[i] == 0:
            # If too close to a ghost, peanlize heavily
            if dist < 2:
                score -= 1300
        else:
            # If the ghost is scared, closer is better
            score += 15.0 / dist
    
    # Number of remaining food pellets
    # slightly penalize for more food pellets
    score -= 5 * len(foodList)
    
    return score

# Abbreviation
better = betterEvaluationFunction
