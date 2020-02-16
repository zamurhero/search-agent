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
        for city, dist in self.roadmap[state]: 
            if city == res:
                return dist

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
    
    # if problem is not Problem:
    #     raise TypeError("First argument must implement abstract class Problem")
    state = problem.result(parent.state, action)
    path_cost = parent.path_cost + problem.step_cost(parent.state, action, state)
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

    def solution(node):
        """ Recursive function to return the sequence of actions that form the 
            solution as a string.
        """
        if node.parent == None:
            return node.state
        else:
            return solution(node.parent) + " - " + node.state

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

    print(recursive_dls(Node(problem.initial_state), problem, limit))


if __name__ == "__main__":
    d = {"albanyNY": [("montreal",226), ("boston",166), ("rochester",148)], 
    "albanyGA": [("tallahassee",120), ("macon",106)], "albuquerque": 
    [("elPaso",267), ("santaFe",61)], "atlanta": [("macon",82), 
    ("chattanooga",117)], "augusta": [("charlotte",161), ("savannah",131)], 
    "austin": [("houston",186), ("sanAntonio",79)], "bakersfield": 
    [("losAngeles",112), ("fresno",107)], "baltimore": [("philadelphia",102), 
    ("washington",45)], "batonRouge": [("lafayette",50), ("newOrleans",80)], 
    "beaumont": [("houston",69), ("lafayette",122)], "boise": [("saltLakeCity",349),
    ("portland",428)], "boston": [("providence",51)], "buffalo": [("toronto",105), 
    ("rochester",64), ("cleveland",191)], "calgary": [("vancouver",605), 
    ("winnipeg",829)], "charlotte": [("greensboro",91)], "chattanooga": 
    [("nashville",129)], "chicago": [("milwaukee",90), ("midland",279)], 
    "cincinnati": [("indianapolis",110), ("dayton",56)], "cleveland": 
    [("pittsburgh",157), ("columbus",142)], "coloradoSprings": [("denver",70), 
    ("santaFe",316)], "columbus": [("dayton",72)], "dallas": [("denver",792), 
    ("mexia",83)], "daytonaBeach": [("jacksonville",92), ("orlando",54)], 
    "denver": [("wichita",523), ("grandJunction",246)], "desMoines": 
    [("omaha",135), ("minneapolis",246)], "elPaso": [("sanAntonio",580), 
    ("tucson",320)], "eugene": [("salem",63), ("medford",165)], "europe": 
    [("philadelphia",3939)], "ftWorth": [("oklahomaCity",209)], "fresno": 
    [("modesto",109)], "grandJunction": [("provo",220)], "greenBay": 
    [("minneapolis",304), ("milwaukee",117)], "greensboro": [("raleigh",74)], 
    "houston": [("mexia",165)], "indianapolis": [("stLouis",246)], 
    "jacksonville": [("savannah",140), ("lakeCity",113)], "japan": 
    [("pointReyes",5131), ("sanLuisObispo",5451)], "kansasCity": [("tulsa",249), 
    ("stLouis",256), ("wichita",190)], "keyWest": [("tampa",446)], "lakeCity": 
    [("tampa",169), ("tallahassee",104)], "laredo": [("sanAntonio",154), 
    ("mexico",741)], "lasVegas": [("losAngeles",275), ("saltLakeCity",486)], 
    "lincoln": [("wichita",277), ("omaha",58)], "littleRock": [("memphis",137), 
    ("tulsa",276)], "losAngeles": [("sanDiego",124), ("sanLuisObispo",182)], 
    "medford": [("redding",150)], "memphis": [("nashville",210)], "miami": 
    [("westPalmBeach",67)], "midland": [("toledo",82)], "minneapolis": 
    [("winnipeg",463)], "modesto": [("stockton",29)], "montreal": [("ottawa",132)],
    "newHaven": [("providence",110), ("stamford",92)], "newOrleans": 
    [("pensacola",268)], "newYork": [("philadelphia",101)], "norfolk": 
    [("richmond",92), ("raleigh",174)], "oakland": [("sanFrancisco",8), 
    ("sanJose",42)], "oklahomaCity": [("tulsa",105)], "orlando": 
    [("westPalmBeach",168), ("tampa",84)], "ottawa": [("toronto",269)], 
    "pensacola": [("tallahassee",120)], "philadelphia": [("pittsburgh",319), 
    ("newYork",101), ("uk1",3548)], "philadelphia": [("uk1",3548)], "phoenix": 
    [("tucson",117), ("yuma",178)], "pointReyes": [("redding",215), 
    ("sacramento",115)], "portland": [("seattle",174), ("salem",47)], 
    "reno": [("saltLakeCity",520), ("sacramento",133)], "richmond": 
    [("washington",105)], "sacramento": [("sanFrancisco",95), ("stockton",51)], 
    "salinas": [("sanJose",31), ("sanLuisObispo",137)], "sanDiego": 
    [("yuma",172)], "saultSteMarie": [("thunderBay",442), ("toronto",436)], 
    "seattle": [("vancouver",115)], "thunderBay": [("winnipeg",440)]}
    new_d = dict()
    for key in d:
        if key not in new_d:
            new_d[key] = []
        for city,dist in d[key]:
            new_d[key].append((city,dist))
            if city not in new_d:
                new_d[city] = [(key,dist)]
            else:
                new_d[city].append((key,dist))
    shortest_path_problem = ShortestPathProblem(new_d, "austin", "houston")
    print(shortest_path_problem.step_cost("austin", "houston", "houston"))
    # depth_limited_search(shortest_path_problem, 1000)