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
        pos = currentGameState.getPacmanPosition()
        food = currentGameState.getFood()
        ghostStates = currentGameState.getGhostStates()
        scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]


        import math
        newGhostPositions = [g.getPosition() for g in newGhostStates]
        newFoodPositions = [(i, j) for i in range(newFood.width) for j in range(newFood.height) if newFood[i][j]]

        if not newFoodPositions:
            return math.inf

        ghostPositions = [g.getPosition() for g in ghostStates]
        foodPositions = [(i, j) for i in range(food.width) for j in range(food.height) if food[i][j]]

        md = util.manhattanDistance;
        GDists = [md(newPos, g) for g in newGhostPositions]
        FDists = [md(newPos, f) for f in newFoodPositions]
        minG = min(GDists)
        minF = min(FDists)


        if minG == 0 or minG == 1:
            return -math.inf

        if len(foodPositions) > len(newFoodPositions):
            return math.inf

        return -minF


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
        class Node:

            def __init__(self, state, agentIndex, depth, action=None):
                self.state = state
                self.agentIndex = agentIndex
                self.depth = depth
                self.children = []
                if self.depth > 0:
                    self.getChildren()
                self.action = action

            def getNumAgents(self):
                return self.state.getNumAgents()

            def generateSuccessor(self, action):
                return self.state.generateSuccessor(self.agentIndex, action)

            def getLegalActions(self):
                return self.state.getLegalActions(self.agentIndex)

            def isMax(self):
                if not self.agentIndex:
                    return True
                return False

            def isLose(self):
                return self.state.isLose()

            def isWin(self):
                return self.state.isWin()

            def getChildren(self):
                for action in self.getLegalActions():
                    successorState = self.generateSuccessor(action)
                    agentIndex = (self.agentIndex + 1) % self.getNumAgents()
                    d = self.depth if agentIndex != 0 else self.depth - 1
                    self.children.append(Node(successorState, agentIndex, d, action))


        def minimax(node, depth, eFunction):
            import math
            if not node.children:
                return eFunction(node.state), node.action


            if node.isMax():
                value = -math.inf
                for child in node.children:
                    mm = minimax(child, depth - 1, eFunction)
                    if value <= mm[0]:
                        value = mm[0]
                        action = child.action

            else:
                value = math.inf
                for child in node.children:
                    mm = minimax(child, depth - 1, eFunction)
                    if value >= mm[0]:
                        value = mm[0]
                        action = child.action

            return (value, action)

        root = Node(gameState, 0, self.depth)
        action = minimax(root, self.depth, self.evaluationFunction)[1]

        return action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        class Node:

            def __init__(self, state, agentIndex, depth, evaluationFunction, alpha, beta, action=None):
                self.state = state
                self.agentIndex = agentIndex
                self.depth = depth
                self.children = []
                self.eF = evaluationFunction
                self.action = action
                self.alpha = alpha
                self.beta = beta

            def getNumAgents(self):
                return self.state.getNumAgents()

            def generateSuccessor(self, action):
                return self.state.generateSuccessor(self.agentIndex, action)

            def getLegalActions(self):
                return self.state.getLegalActions(self.agentIndex)

            def isMax(self):
                if not self.agentIndex:
                    return True
                return False

            def isLose(self):
                return self.state.isLose()

            def isWin(self):
                return self.state.isWin()

            def alphaBetaPrune(self):
                import math
                if self.depth == 0 or self.isLose() or self.isWin():
                    return self.eF(self.state), self.action

                if self.isMax():
                    value = -math.inf
                    for action in self.getLegalActions():
                        agentIndex = (self.agentIndex + 1) % self.getNumAgents()
                        d = self.depth if agentIndex != 0 else self.depth - 1
                        succ = self.generateSuccessor(action)
                        child = Node(succ, agentIndex, d, self.eF, self.alpha, self.beta, action)
                        abp = child.alphaBetaPrune()
                        if value < abp[0]:
                            value = abp[0]
                            childAction = child.action
                            self.alpha = max(self.alpha, value)
                        if self.alpha > self.beta:
                            break;

                else:
                    value = math.inf
                    for action in self.getLegalActions():
                        agentIndex = (self.agentIndex + 1) % self.getNumAgents()
                        d = self.depth if agentIndex != 0 else self.depth - 1
                        succ = self.generateSuccessor(action)
                        child = Node(succ, agentIndex, d, self.eF, self.alpha, self.beta, action)
                        abp = child.alphaBetaPrune()
                        if value > abp[0]:
                            value = abp[0]
                            childAction = child.action
                            self.beta= min(self.beta, value)
                        if self.alpha > self.beta:
                            break;


                return value, childAction

        import math
        root = Node(gameState, 0, self.depth, self.evaluationFunction, -math.inf, math.inf)
        return root.alphaBetaPrune()[1]


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
