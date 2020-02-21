from math import sqrt, cos, pi
from abc import ABC, abstractmethod
from heapq import heappush, heappop
import sys
sys.setrecursionlimit(10000)
from road_data import adj, coords, spherical, euclidean, my_heuristic
from copy import deepcopy
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

class InformedProblem(Problem):
    """ Abstract class for a problem with heuristic.

    Subclass of the Problem class. Has an additional 'heuristic' method to 
    calculate heuristic of getting to the goal state from a given state.
    """

    def heuristic(self, state):
        """ Return the  estimated cost (heuristic) of the cheapest path from 
            the given state to a goal state
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
        return (state == self.goal_state)

    def step_cost(self, state, action, res):
        """ Returns the distance from city 'state' to city 'res' """
        for city, dist in self.roadmap[state]: 
            if city == res:
                return dist

class ShortestPathInformedProblem(InformedProblem, ShortestPathProblem):
    """ Shortest Problem with a heuristic. 
    
    Implements the InformedProblem abstract class and extends the 
    ShortestPathProblem class.

    The attributes listed below are only the ones not included in base class.

    Attributes:
        heuristic_map: A dictionary that stores the estimated cost of getting 
            from each state in the state space to the goal.
        Example:

        {state1: heur_state_1, state2: heur_state_2, state3: heur_state_3}
    """

    def __init__(self, roadmap, initial_state, goal_state, heuristic_map):
        """ Inits the problem with the given attributes """
        super().__init__(roadmap, initial_state, goal_state)
        self.heuristic_map = heuristic_map

    def heuristic(self, state):
        """ Returns the heuristic value of the given state """
        return self.heuristic_map[state]

class Node:
    """ A class to represent a node in a search tree.
    
    Attributes:
        state: the state in the state space to which the node corresponds
        parent: the node in the search tree that generated this node
        action: the action that was applied to the parent to generate the node 
        path_cost: the cost of the path from the initial state to the node
    """
    def __init__(self, state, parent=None, action=None, path_cost=0, 
                 problem=None):
        """ Inits the Node with the provided attributes """
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.problem = problem
        if self.problem:
            self.f = self.path_cost + self.problem.heuristic(self.state)
        else:
            self.f = None

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
    state = problem.result(parent.state, action)
    path_cost = parent.path_cost + problem.step_cost(parent.state, action, 
                                                     state)
    child = Node(state, parent, action, path_cost, problem=parent.problem)
    return child

class HashStack:
    """ A data structure that combines the capabilities of a stack and a hash
    """

    def __init__(self):
        """ Init a set to support efficient membership testing and a list that 
            implements a stack
        """
        self.hash = set()
        self.stack = []

    def is_empty(self):
        """ Returns boolean indicating whether or not stack is empty """
        return not self.stack

    def pop(self):
        """ Returns the last inserted node among current nodes """
        shead = self.stack.pop()
        self.hash.remove(shead.state)
        return shead

    def insert(self, node):
        """ Inserts a node to the end of the stack """
        self.stack.append(node)
        self.hash.add(node.state)

    def __contains__(self, state):
        """ Returns whether or not a state is already in the stack """
        return state in self.hash

    def __len__(self):
        """ Returns the length of the stack """
        return len(self.stack)

class HashQueue:
    """ A data structure that combines the capabilities of a queue and a hash 
        table. 
    """

    def __init__(self):
        """ Init a set to support efficient membership testing and a list that 
            implements a queue
        """
        self.hash = set()
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

    def __contains__(self, state):
        """ Returns whether or not a state is already in the queue """
        return state in self.hash

    def __len__(self):
        """ Returns the length of the queue """
        return len(self.queue)

class HashPriorityQueue:
    """ A data structure that combines the capabilities of a priority queue and 
        a hash map. 
    """
    
    def __init__(self):
        """ Init a dict to support efficient membership testing and a list that 
            implements a priority queue
        """
        self.hash = dict()
        self.queue = []

    def is_empty(self):
        """ Returns boolean indicating whether or not queue is empty """
        return not self.queue

    def pop(self):
        """ Returns node with highest priority """
        priority, node = heappop(self.queue)
        del self.hash[node.state]
        return node

    def insert(self, node, priority):
        """ inserts a node into the queue according to its priority """
        heappush(self.queue, (priority, node))
        self.hash[node.state] = node.path_cost

    def cost(self, state):
        """ Returns bool indicating whether or not a node in the queue has a 
            greater path_cost than the given cost.
        """
        return self.hash[state]

    def replace(self, new_node, priority):
        """ Replaces the node with the given state with the given new_node. 
        """
        for prio, node in self.queue:
            if node.state == new_node.state:
                self.queue.remove((prio, node))
                break
        self.insert(new_node, priority)
                
    def __contains__(self, state):
        """ Returns whether or not a state is already in the queue """
        return state in self.hash

    def __len__(self):
        return len(self.queue)

def solution(node):
    """ Recursive function to return the sequence of actions that form the 
        solution as a string.
    """
    if node.parent == None:
        return node.state
    else:
        return solution(node.parent) + " - " + node.state

def depth_first_search(problem):
    """ Depth First Search 

    An iterative implementation of Depth First Search.

    The iterative version is not memory efficient as it would take O(db) memory
    in the worst case. But it avoids the proliferation of redundant paths.

    Args:
        problem: The Search problem on which to implement Depth Limited Search.
            Must implement the abstract class 'Problem'.

    Returns:
        Prints the following information to terminal if a goal is found:

        - The number of nodes expanded.
        - The maximum size of the LIFO queue (stack) during search.
        - The final path length.
        - The final path represented as a sequence of states.

        If a goal is not found, it returns either "failure".
    """

    node = Node(problem.initial_state)
    if problem.goal_test(node.state):
        return solution(node)
    frontier = HashStack()
    frontier.insert(node)
    explored = set()
    num_nodes_exp = 0
    max_qsize = 1
    while True:
        if frontier.is_empty():
            print("failure")
            return
        node = frontier.pop()
        explored.add(node.state)
        num_nodes_exp += 1
        for action in problem.actions(node.state):
            child = child_node(problem, node, action)
            if child.state not in explored and child.state not in frontier:
                is_exp = True
                if problem.goal_test(child.state):
                    print(num_nodes_exp)
                    print(max_qsize)
                    print(child.path_cost)
                    print(solution(child))
                    return
                frontier.insert(child)
                max_qsize = max(max_qsize, len(frontier))
                

def depth_limited_search(problem, limit):
    """ Depth Limited Search
    
    A function that implements Depth Limited Search in recursive fashion.
    
    To peform Regular Depth First Search on a finite state space, 'limit' 
    should be set to float('inf').

    The space complexity in the worst case for this recursive version is O(bd),
    whereas the iterative version would take O(|E|) ≈ O(bᵈ) memroy in the worst 
    case.
    
    Args:
        problem: The Search problem on which to implement Depth Limited Search.
            Must implement the abstract class 'Problem'.
        limit: Depth limit until which to search the state space.

    Returns:
        Prints the solution as a sequence of states if a goal is found:
        If a goal is not found, it returns either "cutoff" or "failure".
    """

    def recursive_dls(node, problem, limit, expl):
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
            expl.add(node.state)
            cutoff_occured = False
            for action in problem.actions(node.state):
                child = child_node(problem, node, action)
                if child.state not in expl:
                    result = recursive_dls(child, problem, limit - 1, set(expl))
                    if result == "cutoff":
                        cutoff_occured = True
                    elif result != "failure":
                        return result
            if cutoff_occured:
                return "cutoff"
            else:
                return "failure"
    
    explored = set()
    print(recursive_dls(Node(problem.initial_state), problem, limit, explored))

def a_star_search(problem):
    """ Returns a solution or failure """
    node = Node(problem.initial_state)
    frontier = HashPriorityQueue()
    frontier.insert(node, node.path_cost+problem.heuristic(node.state))
    explored = set()
    num_nodes_exp = 0
    max_qsize = 1
    while True:
        if frontier.is_empty():
            print("failure")
            return
        node = frontier.pop()
        if problem.goal_test(node.state):
            print(num_nodes_exp)
            print(max_qsize)
            print(node.path_cost)
            print(solution(node))
            return
        explored.add(node.state)
        num_nodes_exp += 1
        for action in problem.actions(node.state):
            child = child_node(problem, node, action)
            if child.state not in explored and child.state not in frontier:
                frontier.insert(child, 
                                problem.heuristic(child.state)+child.path_cost)
            elif (child.state in frontier and 
                  frontier.cost(child.state) > child.path_cost):
                frontier.replace(child, problem.heuristic(child.state)+
                    child.path_cost)
        max_qsize = max(max_qsize, len(frontier))

def recursive_best_first_search(problem):
    """ Returns a solution, or failure. """

    def rbfs(problem, node, f_limit):
        """ Helper function for recursion.

        Returns a solution, or failure and a new f-cost limit
        """
        if problem.goal_test(node.state):
            return solution(node), -1
        successors = []
        for action in problem.actions(node.state):
            child = child_node(problem, node, action)
            if node.parent==None or child.state != node.parent.state:
                successors.append(child_node(problem, node, action))
        if not successors:
            return ("failure", float('inf'))
        for s in successors:
            s.f = max(s.path_cost + problem.heuristic(s.state), node.f)
        while True:
            min_f = float('inf')
            for s in successors:
                if s.f < min_f:
                    min_f = s.f
                    best = s
            if best.f > f_limit:
                return ("failure", best.f)
            alternative = float('inf')
            for s in successors:
                if s.f < alternative and s.f >= min_f:
                    alternative = s.f
            (result, best.f) = rbfs(problem, best, min(f_limit, alternative))
            if result != "failure":
                return result, best.f

    print(rbfs(problem, Node(problem.initial_state, problem=problem), float('inf'))[0])

if __name__ == "__main__":
    new_d = dict()
    for key in adj:
        if key not in new_d:
            new_d[key] = []
        for city,dist in adj[key]:
            new_d[key].append((city,dist))
            if city not in new_d:
                new_d[city] = [(key,dist)]
            else:
                new_d[city].append((key,dist))
    for i in new_d:
        for j in new_d:
            if i!=j:
                shortest_path_problem = ShortestPathInformedProblem(new_d, i, j,
                                                                    my_heuristic(
                                                                        coords,j, new_d
                                                                    ))
                recursive_best_first_search(shortest_path_problem)
            print("\n")
    # shortest_path_problem = ShortestPathInformedProblem(new_d, "boston", "japan", my_heuristic(coords,"japan",new_d))
    # recursive_best_first_search(shortest_path_problem)