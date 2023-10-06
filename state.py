
class State:
    """This is used as an enumerate to map state variable names 
        to positions in the state representation and the representation dimension.
        e.g. if the state vector contains [x,y,h], the enum 
        maps x->0, y->1, h->2, LENGTH->3
    """
    X = 0
    Y = 1
    HEADING = 2
    LENGTH = 3
