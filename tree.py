#%%
import random
import numpy as np
import pandas as pd
from numbers import Number
from copy import deepcopy
import time
#%%

import requests

def lineNotifyMessage(token, msg):
  headers = {
      "Authorization": "Bearer " + token,
      "Content-Type" : "application/x-www-form-urlencoded"
  }

  payload = {'message': msg}
  r = requests.post("https://notify-api.line.me/api/notify", headers=headers, params=payload)
  return r.status_code
#%%
class GlobalParameter(object):
    def __init__(self):
        self.col_names = ["aa", 'b', 'c']

GLOBAL = GlobalParameter()
def set_global(df):
    '''

    :param df:
    :type df: pd.DataFrame
    :return:
    '''
    setattr(GLOBAL, "df", df)
    col = df.columns.to_list()
    setattr(GLOBAL, "col_names", col)


#%%
## operation setting
def safe_div(a, b):
    return a / b if b else a
def safe_pow(a, b):
    return np.abs(a) ** np.abs(b)
OPERATIONS = (
    {"func": np.add, "arg_count": 2, "format_str": "({} + {})", "test": "+"},
    {"func": np.subtract, "arg_count": 2, "format_str": "({} - {})", "test": "-"},
    {"func": np.multiply, "arg_count": 2, "format_str": "({} * {})", "test": "*"},
    {"func": np.true_divide, "arg_count": 2, "format_str": "({} / {})", "test": "/"},
    {"func": np.float_power, "arg_count": 2, "format_str": "({} ** {})", "test": "**"},
    {"func": np.negative, "arg_count": 1, "format_str": "-({})", "test": "minus"},
)
OPERATIONS_UNI = []
OPERATIONS_BI = []
OPERATIONS_TRI = []
for operator in OPERATIONS:
    if 1 == operator["arg_count"]:
        OPERATIONS_UNI.append(operator)
    elif 2 == operator["arg_count"]:
        OPERATIONS_BI.append(operator)
    else:
        OPERATIONS_TRI.append(operator)

## number setting
NUMBER_INT = range(10)
NUMBERS = []
for i in range(10):
    NUMBERS.append(NUMBER_INT[i])
    if i == 0:
        continue
    NUMBERS.append(NUMBER_INT[i] / 10)
#%%
def nodecheck(node, child):
    '''

    :param child:
    :param node:
    :type node: Node
    :return:
    '''
    if not isinstance(child, Node):
        print("--------------------")
        print("--------------------")
        node.program_print()
        print("--------------------")
        print("--------------------")
    pass
#%%
def evaluate(root, df):
    '''Evaluate value of the tree in given df, in recursive way. Use np to calculate the expression

    :param root: tree root for the expression tree
    :type root: Node
    :param df: dataframe of the features, [col_name, row], numbers for cell
    :type df: DataFrame
    :return: calculated value for this tree
    :rtype: [float]
    '''
    if root.is_leaf():
        if isinstance(root.value, Number):
            return [root.value] * df.shape[0]
        return df[root.value].to_list()
    nodecheck(root, root.left)
    if 2 == root.arg_count:
        nodecheck(root, root.right)
        return root.value["func"](get_leaves(root.left, 0), get_leaves(root.right, 0))
    return root.value["func"](get_leaves(root.left, 0))

#%%
LEFT = 'left'
RIGHT = 'right'
VAL = 'val'
VALUE = 'value'
def _randChildAttr(root):
    '''

    :param root:
    :type root: Node
    :return:
    :rtype: str
    '''
    if root.is_leaf():
        print("-----when try get randChild--------")
        print("-----got a leaf-----")
        root.program_print()
        print("-------------")
        print("-------------")
        return root
    if 2 == root.arg_count:
        attr = random.choice((LEFT, RIGHT))
    else:
        attr = LEFT
    return attr

def _generate_random_internode_count(height):
    """Return a random leaf count for building binary trees.

    :param height: Height of the binary tree.
    :type height: int
    :return: Random leaf count.
    :rtype: int
    """
    if height <= 1:
        return 1
    return random.randint(1, 2**(height-1))
    # max_leaf_count = 2 ** height
    # half_leaf_count = max_leaf_count // 2
    #
    # # A very naive way of mimicking normal distribution
    # roll_1 = random.randint(0, half_leaf_count)
    # roll_2 = random.randint(0, max_leaf_count - half_leaf_count)
    # return roll_1 + roll_2 or half_leaf_count


def _generate_random_internode_values(height):
    """Return random OPERATIONS for building binary trees.

    :param height: Height-1 of the binary tree.
    :type height: int
    :return: Randomly generated OPERATIONS.
    :rtype: [dict]
    """

    max_node_count = int(2 ** (height + 1) - 1)
    node_values = []
    for _ in range(max_node_count):
        node_values.append(random.choice(OPERATIONS))
    if len(node_values) == 0:
        # prevent empty list
        node_values.append(random.choice(OPERATIONS))

    return node_values


def _build_tree_string(root):
    """Recursively walk down the expression tree and build a pretty-print string.

    :param root: Root node of the binary tree.
    :type root: Node
    :return: .
    :rtype: str

    .. _pre-order
    """
    if root.is_leaf():
        return "{}".format(root.value)
    if 2 == root.arg_count:
        return root.value["format_str"].format(_build_tree_string(root.left), _build_tree_string(root.right))
    return root.value["format_str"].format(_build_tree_string(root.left))

    # if root.leaf:
    #     return "{}".format(root.value)
    # nodecheck(root, root.left)
    # if 2 == root.arg_count:
    #     nodecheck(root, root.right)
    #     return root.value["format_str"].format(_build_tree_string(root.left), _build_tree_string(root.right))
    # if 1 == root.arg_count:
    #     return root.value["format_str"].format(_build_tree_string(root.left))

def _get_tree_properties(root):
    """Inspect the binary tree and return its properties (e.g. height).

    :param root: Root node of the binary tree.
    :type root: Node
    :return: Binary tree properties.
    :rtype: dict
    """
    size = 0
    leaf_count = 0
    min_leaf_depth = 0
    max_leaf_depth = -1
    current_level = [root]

    while len(current_level) > 0:
        max_leaf_depth += 1
        next_level = []

        for node in current_level:
            size += 1

            # Node is a leaf.
            if node.left is None and node.right is None:
                if min_leaf_depth == 0:
                    min_leaf_depth = max_leaf_depth
                leaf_count += 1

            if node.left is not None:
                next_level.append(node.left)

            if node.right is not None:
                next_level.append(node.right)

        current_level = next_level

    return {
        'height': max_leaf_depth,
        'size': size,
        'leaf_count': leaf_count,
        'min_leaf_depth': min_leaf_depth,
        'max_leaf_depth': max_leaf_depth,
    }

def _get_internal_nodes(root):
    '''

    :param root:
    :type root: Node
    :return:
    :rtype: [Node]
    '''
    current_level = [root]
    nodes = []
    levels = []
    level = 1

    while len(current_level) > 0:
        next_level = []
        for node in current_level:
            if not node.is_leaf():
                # avoid getting leaves
                nodes.append(node)
                levels.append(level)
            if node.left is not None:
                next_level.append(node.left)
            if node.right is not None:
                next_level.append(node.right)
        level +=1
        current_level = next_level
    return nodes, levels
#%%
def get_leaves(root, col_name):
    '''Evaluate value of the tree in given df, in recursive way. Use np to calculate the expression

    :param col_name:
    :type col_name: list
    :param root: tree root for the expression tree
    :type root: Node
    :return: calculated value for this tree
    :rtype: [float]
    '''
    if root.is_leaf():
        if isinstance(root.value, Number):
            return [root.value] * df.shape[0]
        return df[root.value].to_list()
    nodecheck(root, root.left)
    if 2 == root.arg_count:
        nodecheck(root, root.right)
        return root.value["func"](get_leaves(root.left, col_name), get_leaves(root.right, col_name))
    return root.value["func"](get_leaves(root.left, col_name))
class Node(object):
    """Represents a binary tree node.

    This class provides methods and properties for managing the current node,
    and the binary tree in which the node is the root of. When a docstring in
    this class mentions "symbolic tree", it is referring to the current node
    as well as all its descendants.

    :param value: Node value (must be a parameter or operator)
    :type value: parameter for leaves, operation for internal nodes
    :param leaf: store parameter if leaf, store operation otherwise
    :type leaf: bool(True if leaf)
    :param left: Left child node (default: None)
    :type left: Node
    :param right: Right child node (default: None)
    :type right: Node
    """
    def __init__(self, value, leaf=False, left=None, right=None, parent=None):
        '''

        :param value:
        :param leaf:
        :param left:
        :param right:
        :param parent:
        :type parent: Node
        '''
        self.value = self.val = value
        self.left = left
        self.right = right
        # only use .leaf at tree()
        self.leaf = leaf
        if not leaf:
            self.arg_count = self.value["arg_count"]
        else:
            self.arg_count = 0
        self.parent = parent
        self._program = None

    def is_leaf(self):
        if self.right == self.left is None:
            return True
        return False

    def program_print(self):
        """print program

        """
        if self._program is None:
            self._program = _build_tree_string(self)
        print(self._program)
        return None

    def rand_child(self):
        '''

        :return:
        :rtype: Node
        '''
        return getattr(self, _randChildAttr(self))

    def rand_internal(self):
        '''

        :return:
        :rtype: Node
        '''
        internals, levels = _get_internal_nodes(self)
        rand_idx = random.randrange(0, len(internals))
        return internals[rand_idx], levels[rand_idx]

    @property
    def program_express(self):
        '''

        :return: program expression
        :rtype: str
        '''
        if self._program is None:
            self._program = _build_tree_string(self)
        return self._program

    @property
    def height(self):
        """Return the height of the binary tree.

        Height of a binary tree is the number of edges on the longest path
        between the root node and a leaf node. Binary tree with just a single
        node has a height of 0.

        :return: Height of the binary tree.
        :rtype: int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.left.left = Node(3)
            >>>
            >>> print(root)
            <BLANKLINE>
                1
               /
              2
             /
            3
            <BLANKLINE>
            >>> root.height
            2

        .. note::
            A binary tree with only a root node has a height of 0.
        """
        return _get_tree_properties(self)['height']

    @property
    def size(self):
        """Return the total number of nodes in the binary tree.

        :return: Total number of nodes.
        :rtype: int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.right = Node(4)
            >>>
            >>> root.size
            4

        .. note::
            This method is equivalent to :func:`binarytree.Node.__len__`.
        """
        return _get_tree_properties(self)['size']

    @property
    def leaf_count(self):
        """Return the total number of leaf nodes in the binary tree.

        A leaf node is a node with no child nodes.

        :return: Total number of leaf nodes.
        :rtype: int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.right = Node(4)
            >>>
            >>> root.leaf_count
            2
        """
        return _get_tree_properties(self)['leaf_count']

    @property
    def max_leaf_depth(self):
        """Return the maximum leaf node depth of the binary tree.

        :return: Maximum leaf node depth.
        :rtype: int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.right.left = Node(4)
            >>> root.right.left.left = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
              1____
             /     \\
            2       3
                   /
                  4
                 /
                5
            <BLANKLINE>
            >>> root.max_leaf_depth
            3
        """
        return _get_tree_properties(self)['max_leaf_depth']

    @property
    def properties(self):
        """Return various properties of the binary tree.

        :return: Binary tree properties.
        :rtype: dict

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.left = Node(4)
            >>> root.left.right = Node(5)
            >>> props = root.properties
            >>>
            >>> props['height']         # equivalent to root.height
            2
            >>> props['size']           # equivalent to root.size
            5
            >>> props['max_leaf_depth'] # equivalent to root.max_leaf_depth
            2
            >>> props['min_leaf_depth'] # equivalent to root.min_leaf_depth
            1
            >>> props['leaf_count']     # equivalent to root.leaf_count
            3
        """
        properties = _get_tree_properties(self)

        return properties


#%%
def tree(variables, height=4, depth=0):
    """Generate a random expression tree and return its root node.

    :param variables: names of the variables.
    :type variables: [str]
    :param height: Height of the tree (default: 3, range: 0 - 9 inclusive).
    :type height: int
    :param depth:
    :type depth: int
    :return: Root node of the binary tree.
    :rtype: binarytree.Node

    **Example**:

    .. doctest::

        >>> from binarytree import tree
        >>>
        >>> root = tree()
        >>>
        >>> root.height
        3

    # .. doctest::
    #
    #     >>> from binarytree import tree
    #     >>>
    #     >>> root = tree(height=20)  # doctest: +IGNORE_EXCEPTION_DETAIL
    #     Traceback (most recent call last):
    #      ...
    #     TreeHeightError: height must be an int between 0 - 9
    """


    def _insert_random_leaf_values(root):
        """Preorder traverse tree and insert variable or number at leaf.

        :param root: root of the expression tree.
        :type root: Node
        :return: None
        :rtype: None
        """
        if isinstance(root, Node):
            if 2 == root.arg_count:
                attrs = [LEFT, RIGHT]
            else:
                attrs = [LEFT]
            for attr in attrs:
                operator = random.choice(random.choice((NUMBERS, variables)))
                if getattr(root, attr) is None:
                    setattr(root, attr, Node(value=operator, leaf=True, parent=root))
                else:
                    _insert_random_leaf_values(getattr(root, attr))
        pass

    generating_height = max(int(height - depth), 0)

    if generating_height > 1:
        # print(type(height))
        operations: list
        operations = _generate_random_internode_values(generating_height - 1)
        inter_leaf_count = _generate_random_internode_count(generating_height - 1)
        root = Node(operations.pop(0))
        inter_leaves = 0

        ## grow internal nodes, operation
        for operation in operations:
            node = root
            growing_depth = 0
            inserted = False
            while growing_depth < (generating_height - 1) and not inserted:
                if node.arg_count == 2:
                    attr = random.choice((LEFT, RIGHT))
                else:
                    attr = LEFT
                if getattr(node, attr) is None:
                    setattr(node, attr, Node(value=operation, parent=node))
                    inserted = True
                node = getattr(node, attr)
                growing_depth += 1

            if inserted and growing_depth == (generating_height - 1):
                inter_leaves += 1
            if inter_leaves == inter_leaf_count:
                break
        ## grow leaves
        _insert_random_leaf_values(root)
    else:
        root = Node(value=random.choice(random.choice((NUMBERS, variables))), leaf=True)

    return root
#%%
# ## select_random_node
# def select_random_node(selected, parent, depth):
#     '''
#
#     :param selected:
#     :type selected: Node
#     :param parent:
#     :type parent: Node
#     :param depth:
#     :type depth: int
#     :return:
#     :rtype: Node
#     '''
#     current_level = [selected]
#     result = []
#     result_level = []
#     level = 1
#     while len(current_level) > 0:
#         next_level = []
#         for node in current_level:
#             result.append(node)
#             result_level.append(level)
#             if node.left is not None:
#                 next_level.append(node.left)
#             if node.right is not None:
#                 next_level.append(node.right)
#             level +=1
#         current_level = next_level
#     chosen_ind = random.randrange(0, selected.size)
#     chosen_node = result[chosen_ind]
#     chosen_level = result_level[chosen_ind]
#
#     # print("choose")
#     # choose.program_print()
#     return chosen_node, chosen_level
#     # if 2 == selected.height:
#     #     return parent
#     # # favor nodes near the root
#     # if random.randint(0, 10) < 2*depth:
#     #     return selected
#     # if 1 == selected.arg_count:
#     #     return select_random_node(selected.left, selected, depth+1)
#     # return select_random_node(random.choice((selected.left, selected.right)), selected, depth+1)

#%%
## mutation
def do_mutate(root, col_name, MUTATE_PCT=0.1, version=1):
    '''

    :param MUTATE_PCT:
    :param root:
    :type root: Node
    :return:
    :rtype: Node
    '''
    offspring = deepcopy(root)
    def mutating(node):
        ''' mutate node by right type of node

        :param node:
        :type node: Node
        :return:
        '''
        if node.is_leaf():
            chosen = random.choice(random.choice((NUMBERS, col_name)))
        else:
            if 1 == node.arg_count:
                chosen = random.choice(OPERATIONS_UNI)
            else:
                chosen = random.choice(OPERATIONS_BI)
        setattr(node, "value", chosen)
        return node

    if 1 == version:
        mutate_parent, parent_level = offspring.rand_internal()

        setattr(mutate_parent, _randChildAttr(mutate_parent), tree(col_name, depth=parent_level + 1))
    else:
        result = []
        stack = [offspring]

        while stack:
            node = stack.pop()
            if node:
                result.append(node)
                if random.random() < MUTATE_PCT:
                    node = mutating(node)
                stack.append(node.right)
                stack.append(node.left)

    return offspring



#%%
## crossover
def do_xover(selected1, selected2, version=1):
    '''

    :param selected1:
    :type selected1: Node
    :param selected2:
    :type selected2: Node
    :return:
    :rtype: Node
    '''
    if 1 == version:
        offspring = deepcopy(selected1)
        # print(offspring.program_print())
        # xover_parent1 = select_random_node(offspring, None, 0)
        # print(xover_parent1.program_print())
        # print("\np2\n")
        # print(selected2.program_print())
        xover_parent1, p1_level = offspring.rand_internal()
        xover_parent2, p2_level = selected2.rand_internal()

        setattr(xover_parent1, _randChildAttr(xover_parent1), xover_parent2.rand_child())
        # if 2 == xover_parent1.arg_count:
        #     attr1 = random.choice((LEFT, RIGHT))
        # else:
        #     attr1 = LEFT
        # setattr(xover_parent1, attr1, xover_parent2.rand_child())
        return offspring
    else:
        offspring1 = deepcopy(selected1)
        offspring2 = deepcopy(selected2)
        xover_parent1, p1_level = offspring1.rand_internal()
        xover_parent2, p2_level = offspring2.rand_internal()
        x_point1_LR = _randChildAttr(xover_parent1)
        x_point2_LR = _randChildAttr(xover_parent2)
        x_point1 = deepcopy(getattr(xover_parent1, x_point1_LR))
        x_point2 = deepcopy(getattr(xover_parent2, x_point2_LR))
        setattr(xover_parent1, x_point1_LR, x_point2)
        setattr(xover_parent2, x_point2_LR, x_point1)
        return offspring1, offspring2

#%%
class Ranking:
    def __init__(self, prog, fitness):
        '''

        :param prog:
        :type prog: Node
        :param fitness:
        :type fitness: float
        '''
        self.prog = prog
        self.fitness = fitness
#%%
def compare_fitness(ranking):
    '''

    :param ranking:
    :type ranking: Ranking
    :return:
    '''
    return ranking.fitness

#%%
def get_random_root(population, fitness=False, POP_SIZE=False, TOURNAMENT_SIZE=3):
    if fitness:
        # randomly select population members for the tournament
        tournament_members = [
            random.randint(0, POP_SIZE - 1) for _ in range(TOURNAMENT_SIZE)]
        # select tournament member with best fitness
        member_fitness = [(fitness[i], population[i]) for i in tournament_members]
        # print("pick: ")
        # min(member_fitness, key=lambda x: x[0])[1].program_print()

        return min(member_fitness, key=lambda x: x[0])[1]


    # OR randomly select population members
    return random.choice(population)

## get_offspring
## TODO: XOVER_PCT into fun.
def get_offspring(population, fitness, POP_SIZE, col_name,TOURNAMENT_SIZE=3, XOVER_PCT=0.7, version=1):
    '''

    :param population:
    :param fitness:
    :param TOURNAMENT_SIZE:
    :param XOVER_PCT:
    :param POP_SIZE:
    :param col_name:
    :param version:
    :return:
    '''
    parent1 = get_random_root(population, fitness, POP_SIZE, TOURNAMENT_SIZE)
    parent2 = get_random_root(population, fitness, POP_SIZE, TOURNAMENT_SIZE)
    offsprings = [None, None]
    if 1 == version:
        if random.random() < XOVER_PCT:
            # print("do xover")
            return do_xover(parent1, parent2)
        else:
            # print("do mutate")
            return do_mutate(parent1, col_name)

    elif 1.1 == version:
        # parent1 = get_random_root(population, fitness, TOURNAMENT_SIZE, POP_SIZE)
        # parent2 = get_random_root(population, fitness, TOURNAMENT_SIZE, POP_SIZE)
        offsprings[0], offsprings[1] = do_xover(parent1, parent2, version=version)
        for i in range(2):
            offsprings[i] = do_mutate(offsprings[i], col_name, version=version)
        return offsprings[0], offsprings[1]

    elif 1.2 == version:
        parent1 = get_random_root(population)
        parent2 = get_random_root(population)
        offsprings[0], offsprings[1] = do_xover(parent1, parent2, version=version)
        for i in range(2):
            offsprings[i] = do_mutate(offsprings[i], col_name, version=version)
        return offsprings[0], offsprings[1]

def selection(population, offsprings, fitness_pop, fitness_off, POP_SIZE, method="WHEEL"):
    '''return equal amount of POP_SIZE in population and offsprings

    :param population: list of Node
    :type population: list
    :param offsprings: list of Node
    :type offsprings: list
    :param fitness_pop:
    :param fitness_off:
    :param POP_SIZE:
    :return:
    :rtype: list
    '''
    mixed = population + offsprings
    fitness_mixed = fitness_pop + fitness_off
    selected = []
    if "WHEEL" == method:
        raw = []
        for i in range(len(mixed)):
            raw.append(Ranking(mixed[i], fitness_mixed[i]))
        ascending = sorted(raw, key=compare_fitness)
        descending = sorted(raw, key=compare_fitness, reverse=True)
        full_fitness = np.sum(fitness_mixed)
        weight = []
        ascending_prog = []
        for i in range(len(mixed)):
            ascending_prog.append(ascending[i].prog)
            weight.append(round(descending[i].fitness / full_fitness, 3))
        selected = random.choices(ascending_prog, weights=weight, k=POP_SIZE)
    else:
        for _ in range(POP_SIZE):
            selected.append(get_random_root(population=mixed, fitness=fitness_mixed, POP_SIZE=POP_SIZE))
    return selected

def compute_fitness(root, prediction, REG_STRENGTH, y_true):
    '''

        :param root:
        :type root: Node
        :param prediction:
        :type prediction: list
        :param REG_STRENGTH:
        :param y_true:
        :type y_true: dataframe
        :return:
        :rtype: float
        '''
    # print(np.subtract(prediction, y_true.to_list()))
    # print(np.subtract(prediction, y_true.to_list()) ** 2)
    mse = np.average(np.subtract(prediction, y_true.to_list()) ** 2)
    # penalty = root.size ** REG_STRENGTH
    # if "x" not in render_prog(program) and "y" not in render_prog(program):
    #    penalty = 101
    if "x" not in root.program_express and "y" not in root.program_express:
        penalty = 10
    else:
        penalty = 1
    return mse
#%%
if __name__ == '__main__':

    col_name = ['a', 'b', 'c']
    df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [0.7, 0.8, 0.9], [0.2, 0.3, 1.1]]),
                       columns=col_name)
    tt = tree(col_name, height=3)
    tt.program_print()
    #%%
    for _ in range(100):
        t2 = do_mutate(tt, )
        t2.program_print()
    #%%
    col_name = ['a', 'b', 'c']
    df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [0.7, 0.8, 0.9], [0.2, 0.3, 1.1]]),
                      columns=col_name)
    tt = tree(col_name, height=3)
    t1 = tree(col_name, height=3)
    #%%
    tt.program_print()
    t1.program_print()

    #%%
    for _ in range(10):
        t2 = do_xover(tt, t1)
        t2.program_print()
    ## get_random_parent
    #%%
    col_name = ['a', 'b', 'c']
    df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [0.7, 0.8, 0.9], [0.2, 0.3, 1.1]]),
                       columns=col_name)
    tt = tree(col_name, height=3)

    #%%
    temp2 = []
    temp2=get_leaves(tt, 0)

    print(df)
    tt.program_print()
    print(temp2)
    #%%
    print(temp2[1])




