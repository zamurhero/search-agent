from abc import ABC, abstractmethod
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
        pass

    @abstractmethod
    def result(self, state, action):
        """ The result of performing an action on a state
        
        Args:
            state: Instance of self.State
            action: Action performed on state 
        
        Returns:
            Resulting state
        """
        pass

    @abstractmethod
    def goal_test(self, state):
        """ Check if a state is a goal state
        
        Args:
            state: Instance of self.State
        
        Returns:
            Boolean indicating whether or not the given state is a goal state
        """
        pass

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
        pass

class ShortestPathProblem(Problem):
    """ Shortest Path Problem.
    
    This class represents the problem of finding the shortest path between two cities.
    Implements the Abstract class 'Problem'.

    An environment state in this problem is just the name of a city.
    """

    def __init__(self, map, initial_state, goal_state):
        """ Init the problem with a road map and initial state
        
        Args:
            map: A dict that indicates adjacent cities of each city and their distances
                For example:
                
                {'austin': [('houston',186), ('sanAntonio',79)],
                 'buffalo': [('toronto', 105), ('rochester', 64), ('cleveland', 191)]
                 'dallas': [('denver', 792), ('mexia', 83)]}
                 
            initial_state: The city agent is currently located at.
                Example:
                
                'portlandOR'

            goal_state: The destination city.
                Example:

                'bostonMA'
        """
        self.map = map
        self.initial_state = initial_state
        self.goal_state = goal_state

    def actions(self, state):
        """ Returns cities reachable from 'state' """
        return [city for city,_ in self.map[state]]

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
        """ Returns boolean indicating whether given 'state' is the destination """
        return (state == self.goal_test)

    def step_cost(self, state, action, res):
        """ Returns the distance from city 'state' to city 'res' """
        return [dist for city,dist in self.map[state] if city == state ]

class Node:
    """ A class to represent a node in a search tree.
    
    Attributes:
        state: the state in the state space to which the node corresponds
        parent: the node in the search tree that generated this node
        action: the action that was applied to the parent to generate the node 
        path_cost: the cost of the path from the initial state to the node
    """

    def __init__(self,state,parent,action,path_cost):
        """ Inits the Node with the provided attributes """
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost