import random, util, time
from game import Agent
import ghostAgents, pacman
import math

#     ********* Reflex agent- sections a and b *********

# util functions

def furthest(pos, lst):
  furthestPos = None
  furthestDist = -1
  for tup in list(map(lambda idx: (idx, util.manhattanDistance(idx, pos)), lst)):
    (furthestPos, furthestDist) = tup if tup[1] > furthestDist else (furthestPos, furthestDist)
  return furthestPos

def bfs(grid, fromPos, toPos, length):
  if fromPos == toPos:
    return length
  if fromPos[0] < 0 or fromPos[0] >= grid.width or fromPos[0] < 0 or fromPos[0] >= grid.height\
          or grid[fromPos[0]][fromPos[1]]:
    return math.inf
  grid[fromPos[0]][fromPos[1]] = True
  return min([bfs(grid, (fromPos[0] + 1, fromPos[1]), toPos, length + 1),
              bfs(grid, (fromPos[0] - 1, fromPos[1]), toPos, length + 1),
              bfs(grid, (fromPos[0], fromPos[1] + 1), toPos, length + 1),
              bfs(grid, (fromPos[0], fromPos[1] - 1), toPos, length + 1)])


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [(action ,self.evaluationFunction(gameState, action)) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    for score in scores:
      print(score)
    print("chose " + str(legalMoves[chosenIndex]) + "\n\n")
    time.sleep(1)

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
  return gameState.getScore()

######################################################################################
# b: implementing a better heuristic function


def betterEvaluationFunction(gameState):
  """

  The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

  A GameState specifies the full game state, including the food, capsules, agent configurations and more.
  Following are a few of the helper methods that you can use to query a GameState object to gather information about
  the present state of Pac-Man, the ghosts and the maze:

  gameState.getLegalActions():
  gameState.getPacmanState():
  gameState.getGhostStates():
  gameState.getNumAgents():
  gameState.getScore():
  The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
  """

  pacman_pos = gameState.getPacmanPosition()
  food = gameState.getFood().asList()
  diagonal = 4 * max(gameState.getFood().height, gameState.getFood().width)

  furthest_food_pos = furthest(pacman_pos, food)
  biggest_dist_food = furthest(furthest_food_pos, food)
  dist_from_biggest_dist_food = bfs(gameState.getWalls().deepCopy(), pacman_pos, biggest_dist_food, 0)\
    if biggest_dist_food else 0
  # print("pac is in " + str(pacman_pos))
  # print("food is in " + str(biggest_dist_food))
  # print("bfs returned " + str(dist_from_biggest_dist_food))
  # time.sleep(3)
  return gameState.getScore() + ((diagonal - dist_from_biggest_dist_food) / diagonal) * 10

#     ********* MultiAgent Search Agents- sections c,d,e,f*********


class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

  def expectimax(self, gameState, depth, ghostType):

    if gameState.isLose() or gameState.isWin():
        return gameState.getScore()

    if depth == gameState.getNumAgents() * self.depth:
        return self.evaluationFunction(gameState)

    agentIndex = (depth % gameState.getNumAgents())
    isMax = (agentIndex == 0)
    values = [self.expectimax(gameState.generateSuccessor(agentIndex, action), depth + 1, ghostType) for action in
              gameState.getLegalActions(agentIndex)]
    ghost = ghostAgents.util.lookup(ghostType, globals())(agentIndex)
    if isMax:
      return max(values)
    else:
      dist = ghost.getDistribution(gameState)
      distList = None if isMax else [dist[action] for action in gameState.getLegalActions(agentIndex)]
      return sum([distList[i]*values[i] for i in range(len(values))])

######################################################################################
# c: implementing minimax



class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent
  """

  def miniMax(self, gameState, depth):
    if gameState.isLose() or gameState.isWin():
      return gameState.getScore()

    if depth == gameState.getNumAgents() * self.depth:
      return self.evaluationFunction(gameState)

    agentIndex = (depth % gameState.getNumAgents())
    isMax = (agentIndex == 0)
    values = [self.miniMax(gameState.generateSuccessor(agentIndex, action), depth+1) for action in gameState.getLegalActions(agentIndex)]
    return max(values) if isMax else min(values)


  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.miniMax(gameState.generateSuccessor(0, action), 1) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalMoves[chosenIndex]

    # raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning
  """

  def alphaBeta(self, gameState, depth, alpha, beta):
    if gameState.isLose() or gameState.isWin():
      return gameState.getScore()

    if depth == gameState.getNumAgents() * self.depth:
      return self.evaluationFunction(gameState)

    agentIndex = (depth % gameState.getNumAgents())
    if agentIndex == 0:
      currMax = -math.inf
      for action in gameState.getLegalActions(agentIndex):
        v = self.alphaBeta(gameState.generateSuccessor(agentIndex, action), depth + 1, alpha, beta)
        currMax = max(v, currMax)
        alpha = max(currMax, alpha)
        if currMax >= beta:
          return math.inf
      return currMax
    else:
      currMin = math.inf
      for action in gameState.getLegalActions(agentIndex):
        v = self.alphaBeta(gameState.generateSuccessor(agentIndex, action), depth + 1, alpha, beta)
        currMin = min(v, currMin)
        beta = min(currMin, beta)
        if currMin <= alpha:
          return -math.inf
      return currMin


  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.alphaBeta(gameState.generateSuccessor(0, action), 1, -math.inf, math.inf) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    return legalMoves[chosenIndex]

    # BEGIN_YOUR_CODE
    # raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """
  RANDOM = 'ghostAgents.RandomGhost'

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """

    # BEGIN_YOUR_CODE
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.expectimax(gameState.generateSuccessor(0, action), 1, self.RANDOM) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalMoves[chosenIndex]
    # raise Exception("Not implemented yet")
    # END_YOUR_CODE



######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """
  DIRECTIONAL = 'ghostAgents.DirectionalGhost'

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """

    # BEGIN_YOUR_CODE
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.expectimax(gameState.generateSuccessor(0, action), 1, self.DIRECTIONAL) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    return legalMoves[chosenIndex]

    # raise Exception("Not implemented yet")
    # END_YOUR_CODE


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
  """
    Your competition agent
  """

  def getAction(self, gameState):
    """
      Returns the action using self.depth and self.evaluationFunction

    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE



