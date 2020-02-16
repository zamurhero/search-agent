from abc import ABC, abstractmethod
from heapq import heappush, heappop
class Problem(ABC):
    """ Abstract class for a Search Problem.

    Direct instantiation not possible.
    Create a subclass and override the abstact methods before instantiating.
    """

    def initial_state(self):
        """ Returns the initial state of the problem environment """
        return self.initial_state

    @abstractmethod
    def actions(self, state):
        """ The available actions given a state
        
        Args:
            state: Instance of self.State
            
        Returns:
            A list of actions
        """

    @abstractmethod
    def result(self, state, action):
        """ The result of performing an action on a state
        
        Args:
            state: Instance of self.State
            action: Action performed on state 
        
        Returns:
            Resulting state
        """

    @abstractmethod
    def goal_test(self, state):
        """ Check if a state is a goal state
        
        Args:
            state: Instance of self.State
        
        Returns:
            Boolean indicating whether or not the given state is a goal state
        """

    @abstractmethod
    def step_cost(self, state, action, res):
        """ Step cost of going from a state to another thru an action 
        
        Args:
            state: Instance of self.State, state on which action was performed.
            action: The action performed on state.
            res: The state resulted from performing action on state.
        
        Returns:
            An instance of a numeric data type indicating the step cost.
        """

class ShortestPathProblem(Problem):
    """ Shortest Path Problem.
    
    This class represents the problem of finding the shortest path between two 
    cities.
    Implements the Abstract class 'Problem'.

    An environment state in this problem is just the name of a city.

    Attributes:
        roadmap: A dict that indicates adjacent cities of each city and their 
            distances
            For example:
            
            {'austin': [('houston',186), ('sanAntonio',79)],
             'buffalo': [('toronto', 105), ('rochester', 64), ('cleveland', 19)]
             'dallas': [('denver', 792), ('mexia', 83)]}
                
        initial_state: The city agent is currently located at.
            Example:
            
            'portlandOR'

        goal_state: The destination city.
            Example:

            'bostonMA'
    """

    def __init__(self, roadmap, initial_state, goal_state):
        """ Init the ShortestPathProblem with the provided attributes"""
        self.roadmap = roadmap
        self.initial_state = initial_state
        self.goal_state = goal_state

    def actions(self, state):
        """ Returns cities reachable from 'state' """
        return [city for city, _ in self.roadmap[state]]

    def result(self, state, action):
        """ The resulting state from performing 'action' in city 'state'
        
        Simply returns 'action' since performing an action here is nothing
        but travelling to that city.

        For example:

        Performing action 'boston' when in city 'sanFransisco' just takes us
        from 'boston' to 'sanFransisco'. So the result is the action itself.
        """
        return action

    def goal_test(self, state):
        """ Returns boolean indicating whether given 'state' is the 
            destination 
        """
        return (state == self.goal_test)

    def step_cost(self, state, action, res):
        """ Returns the distance from city 'state' to city 'res' """
        return [dist for city, dist in self.roadmap[state] if city == state]

class Node:
    """ A class to represent a node in a search tree.
    
    Attributes:
        state: the state in the state space to which the node corresponds
        parent: the node in the search tree that generated this node
        action: the action that was applied to the parent to generate the node 
        path_cost: the cost of the path from the initial state to the node
    """

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """ Inits the Node with the provided attributes """
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

def child_node(problem, parent, action):
    """ Given parent node, child node.
    
    Takes a parent node and an action and returns the resulting child node.

    Args:
        problem: the problem being solved by the search algorithm/agent
        parent: the parent node
        action: the action performed on the parent node

    Returns:
        the resulting child node
    """
    
    if problem is not Problem:
        raise TypeError("First argument must implement abstract class Problem")
    state = problem.result(parent.state, action)
    path_cost = parent.path_cost + problem.step_cost(parent.state, action)
    child = Node(state, parent, action, path_cost)
    return child

class HashQueue:
    """ A data structure that combines the capabilities of a queue and a hash 
        table. 
    """

    def __init__(self):
        """ Init a set to support efficient membership testing and a list that 
            implements a queue
        """
        self.hash = {}
        self.queue = []

    def is_empty(self):
        """ Returns boolean indicating whether or not queue is empty """
        return not self.queue

    def pop(self):
        """ Returns the earliest inserted node among current nodes """
        qhead = self.queue.pop(0)
        self.hash.remove(qhead.state)
        return qhead

    def insert(self, node):
        """ Inserts a node to the end of the queue """
        self.queue.append(node)
        self.hash.add(node.state)

    def contains(self, state):
        """ Returns whether or not a state is already in the queue """
        return state in self.hash

class HashPriorityQueue:
    """ A data structure that combines the capabilities of a priority queue and 
        a hash map. 
    """
    
    def __init__(self):
        """ Init a dict to support efficient membership testing and a list that 
            implements a priority queue
        """
        self.hash = {}
        self.queue = []

    def is_empty(self):
        """ Returns boolean indicating whether or not queue is empty """
        return not self.queue

    def pop(self):
        """ Returns node with highest priority """
        negative_priority, node = heappop(self.queue)
        del self.hash[node.state]
        return node

    def insert(self, node, priority):
        """ inserts a node into the queue according to its priority """
        heappush(self.queue, (-priority, node)) #negative priority since minheap
        self.hash[node.state] = node.path_cost

    def contains(self, state):
        """ Returns whether or not a state is already in the queue """
        return state in self.hash

def depth_limited_search(problem, limit):
    """ Depth Limited Search
    
    A function that implements Depth Limited Search in recursive fashion.
    
    To peform Regular Depth First Search on a finite state space, 'limit' 
    should be set to float('inf') 
    
    Args:
        problem: The Search problem on which to implement Depth Limited Search.
            Must implement the abstract class 'Problem'.
        limit: Depth limit until which to search the state space.

    Returns:
        Prints the following information to terminal if a goal is found:

        - The number of nodes expanded.
        - The maximum size of the queue during search.
        - The final path length.
        - The final path represented as a sequence of cities.

        If a goal is not found, it returns either "cutoff" or "failure".
    """

    def recursive_dls(node, problem, limit):
        """ A helper function for performing DLS recursively 
        
        Args:
            node: The node from which to perform DLS
            problem: ...
            limit: ...
        """

        if problem.goal_test(node.state):
            return solution(node)
        elif limit == 0:
            return "cutoff"
        else:
            cutoff_occured = False
            for action in problem.actions(node.state):
                child = child_node(problem, node, action)
                result = recursive_dls(child, problem, limit - 1)
                if result == "cutoff":
                    cutoff_occured = True
                elif result != "failure":
                    return result
            if cutoff_occured:
                return "cutoff"
            else:
                return "failure"

    print(resursive_dls(Node(problem.initial_state), problem, limit))