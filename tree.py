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
  r = requests.post("https://notify-api.line.me/api/notify", headers = headers, params = payload)
  return r.status_code

#%%
## operation setting
def safe_div(a, b):
    return a / b if b else a
def safe_pow(a, b):
    return np.abs(a) ** np.abs(b)
operations = (
    {"func": np.add, "arg_count": 2, "format_str": "({} + {})", "test": "+"},
    {"func": np.subtract, "arg_count": 2, "format_str": "({} - {})", "test": "-"},
    {"func": np.multiply, "arg_count": 2, "format_str": "({} * {})", "test": "*"},
    {"func": np.true_divide, "arg_count": 2, "format_str": "({} / {})", "test": "/"},
    {"func": np.float_power, "arg_count": 2, "format_str": "({} ** {})", "test": "**"},
    {"func": np.negative, "arg_count": 1, "format_str": "-({})", "test": "minus"},
)
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
        return root.value["func"](evaluate(root.left, df), evaluate(root.right, df))
    return root.value["func"](evaluate(root.left, df))

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
    max_leaf_count = 2 ** height
    half_leaf_count = max_leaf_count // 2

    # A very naive way of mimicking normal distribution
    roll_1 = random.randint(0, half_leaf_count)
    roll_2 = random.randint(0, max_leaf_count - half_leaf_count)
    return roll_1 + roll_2 or half_leaf_count


def _generate_random_internode_values(height):
    """Return random operations for building binary trees.

    :param height: Height-1 of the binary tree.
    :type height: int
    :return: Randomly generated operations.
    :rtype: [dict]
    """

    max_node_count = int(2 ** (height + 1) - 1)
    node_values = []
    for _ in range(max_node_count):
        node_values.append(random.choice(operations))
    if len(node_values) == 0:
        # prevent empty list
        node_values.append(random.choice(operations))

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

    while len(current_level) > 0:
        next_level = []
        for node in current_level:
            if not node.is_leaf():
                # avoid getting leaves
                nodes.append(node)
            if node.left is not None:
                next_level.append(node.left)
            if node.right is not None:
                next_level.append(node.right)
        current_level = next_level
    return nodes

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
    def __init__(self, value, leaf=False, left=None, right=None):

        self.value = self.val = value
        self.left = left
        self.right = right
        self.leaf = leaf # only use at tree()
        if not leaf:
            self.arg_count = self.value["arg_count"]
        else:
            self.arg_count = 0

    def is_leaf(self):
        if self.right == self.left is None:
            return True
        return False

    def program_print(self):
        """print program
        """
        program = _build_tree_string(self)
        print(program)

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
        internals = _get_internal_nodes(self)

        return random.choice(internals)

    @property
    def program_express(self):
        '''

        :return: program expression
        :rtype: str
        '''
        program = _build_tree_string(self)
        return program

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
    ## number setting
    number_int = range(10)
    numbers = []
    for i in range(10):
        numbers.append(number_int[i])
        if i == 0:
            continue
        numbers.append(number_int[i] / 10)

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
                chosen = random.choice(random.choice((numbers, variables)))
                if None == getattr(root, attr):
                    setattr(root, attr, Node(chosen, leaf=True))
                else:
                    _insert_random_leaf_values(getattr(root, attr))
        pass

    height = int(height-depth)

    # print(type(height))
    operations: list
    operations = _generate_random_internode_values(height - 1)
    inter_leaf_count = _generate_random_internode_count(height - 1)
    root = Node(operations.pop(0))
    leaves = 0

    ## grow internal nodes, operation
    for operation in operations:
        node = root
        depth = 0
        inserted = False
        while depth < (height-1) and not inserted:
            if node.arg_count == 2:
                attr = random.choice((LEFT, RIGHT))
            else:
                attr = LEFT
            if getattr(node, attr) is None:
                setattr(node, attr, Node(operation))
                inserted = True
            node = getattr(node, attr)
            depth += 1

        if inserted and depth == (height-1):
            leaves += 1
        if leaves == inter_leaf_count:
            break


    ## grow leaves
    _insert_random_leaf_values(root)

    return root
#%%
## select_random_node
def select_random_node(selected, parent, depth):
    '''

    :param selected:
    :type selected: Node
    :param parent:
    :type parent: Node
    :param depth:
    :type depth: int
    :return:
    :rtype: Node
    '''
    current_level = [selected]
    result = []

    while len(current_level) > 0:
        next_level = []
        for node in current_level:
            result.append(node)
            if node.left is not None:
                next_level.append(node.left)
            if node.right is not None:
                next_level.append(node.right)
        current_level = next_level
    choose = result[random.randrange(0, selected.size)]
    print("choose")
    choose.program_print()
    return choose
    # if 2 == selected.height:
    #     return parent
    # # favor nodes near the root
    # if random.randint(0, 10) < 2*depth:
    #     return selected
    # if 1 == selected.arg_count:
    #     return select_random_node(selected.left, selected, depth+1)
    # return select_random_node(random.choice((selected.left, selected.right)), selected, depth+1)

#%%
## mutation
def do_mutate(root):
    '''

    :param root:
    :type root: Node
    :return:
    :rtype: Node
    '''
    offspring = deepcopy(root)
    mutate_parent = offspring.rand_internal()

    setattr(mutate_parent, _randChildAttr(mutate_parent), tree(col_name, depth=mutate_parent.height -1))
    return offspring

#%%
col_name = ['a', 'b', 'c']
df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [0.7, 0.8, 0.9], [0.2, 0.3, 1.1]]),
                   columns=col_name)
tt = tree(col_name, height=3)
tt.program_print()
#%%
for _ in range(100):
    t2 = do_mutate(tt)
    t2.program_print()


#%%
## crossover
def do_xover(selected1, selected2):
    '''

    :param selected1:
    :type selected1: Node
    :param selected2:
    :type selected2: Node
    :return:
    :rtype: Node
    '''
    offspring = deepcopy(selected1)
    # print(offspring.program_print())
    # xover_parent1 = select_random_node(offspring, None, 0)
    # print(xover_parent1.program_print())
    # print("\np2\n")
    # print(selected2.program_print())
    xover_parent1 = offspring.rand_internal()
    xover_parent2 = selected2.rand_internal()

    setattr(xover_parent1, _randChildAttr(xover_parent1), xover_parent2.rand_child())
    # if 2 == xover_parent1.arg_count:
    #     attr1 = random.choice((LEFT, RIGHT))
    # else:
    #     attr1 = LEFT
    # setattr(xover_parent1, attr1, xover_parent2.rand_child())
    return offspring
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
def get_random_root(population, fitness, TOURNAMENT_SIZE):
    # randomly select population members for the tournament
    tournament_members = [
        random.randint(0, POP_SIZE - 1) for _ in range(TOURNAMENT_SIZE)]
    # select tournament member with best fitness
    member_fitness = [(fitness[i], population[i]) for i in tournament_members]
    # print("pick: ")
    # min(member_fitness, key=lambda x: x[0])[1].program_print()
    return min(member_fitness, key=lambda x: x[0])[1]

## get_offspring
## TODO: XOVER_PCT into fun.
def get_offspring(population, fitness, TOURNAMENT_SIZE):
    parent1 = get_random_root(population, fitness, TOURNAMENT_SIZE)
    if random.random() < XOVER_PCT:
        parent2 = get_random_root(population, fitness, TOURNAMENT_SIZE)
        # print("do xover")
        return do_xover(parent1, parent2)
    else:
        # print("do mutate")
        return do_mutate(parent1)

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
    return mse, mse *root.size *penalty
#%%
col_name = ['a', 'b', 'c']
df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [0.7, 0.8, 0.9], [0.2, 0.3, 1.1]]),
                   columns=col_name)
tt = tree(col_name, height=3)

#%%
temp2 = []
temp2=evaluate(tt, df)

print(df)
tt.program_print()
print(temp2)
#%%
print(temp2[1])

# %%

## data generate

np.random.seed(0)

n = 10
input_range_low = 100
input_range_up = 200
output_range_low = 20
output_range_up = 500


def generate_uniform_data(lower, upper, n, col=1):
    return np.random.uniform(lower, upper, n * col).reshape(n, col)


x = generate_uniform_data(input_range_low, input_range_up, n, 2)
y = generate_uniform_data(output_range_low, output_range_up, n, 2)

## 製作 slope
slope = (y.T[1] / y.T[0])

## 參數設定
coe_x1x2 = 1.1
pow_x1 = 0.3
pow_x2 = 0.4
coe_y1_eff = (-1)

y1_eff = coe_x1x2 * (x.T[0] ** pow_x1) * (x.T[1] ** pow_x2) / (slope + 1)
y2_eff = coe_x1x2 * (x.T[0] ** pow_x1) * (x.T[1] ** pow_x2) + coe_y1_eff * y1_eff
y2_eff =  2 * (x.T[0] * x.T[1]  )# -  y1_eff
# y2_eff = x.T[0] ** 0.3#* x.T[1]  )# -  y1_eff

y1_eff = y1_eff.reshape(n, 1)

## round
y2_eff = np.round(y2_eff, 2)
y1_eff = np.round(y1_eff, 2)

x = np.round(x, 2)


# %%

## data for SR
d = {'y2': y2_eff.T,
     'x_1': x.T[0],
     'x_2': x.T[1],
     'y_1': y1_eff.T[0],
     }

data = pd.DataFrame(data=d)
print(data.head())
y_true = data.pop("y2")
print(data.head())
print(data.columns.to_list())
# %%

### main
# seed(SEED)
POP_SIZE = 500
# NEW_POP_PCT = 0.1
col_name = data.columns.to_list()
population = [tree(col_name) for _ in range(POP_SIZE)]
#%%
## TODO: revise old program
TOURNAMENT_SIZE = 3
XOVER_PCT = 0.7
REG_STRENGTH = 2
MAX_GENERATIONS = 2000
select_depth = 10

pen_global_best = global_best = float("inf")
unchange_score = 0
ts = time.time()
for gen in range(MAX_GENERATIONS):
    t1 = time.time()
    fitness = []
    change_flag = 1
    kk=0
    for prog in population:
        # print(kk)
        kk += 1
        # prog.program_print()
        prediction = [
            evaluate(prog, data)]
        # print(type(prediction))
        score, pen_score = compute_fitness(prog, prediction, REG_STRENGTH, y_true)
        # print(score)
        if np.isnan(score):
            score = pen_score = np.inf
        fitness.append(score)
        if np.isinf(score):
            continue

        if pen_score < pen_global_best:
            pen_global_best = pen_score
            global_best = score
            best_pred = prediction
            best_prog = prog
            change_flag = 0

    if change_flag:
        unchange_score = unchange_score + 1
    else:
        unchange_score = 0
    # print(unchange_score)

    prog_express = best_prog.program_express
    t2 = time.time()
    print(
        "\nunchange_score: %d\nGeneration: %d\nBest Score: %.2f\nPen Best Score: %.2f\nMedian score: %.2f\nBest program: %s\nTime used: %d sec\n"
        % (
            unchange_score,
            gen,
            global_best,
            pen_global_best,
            pd.Series(fitness).median(),
            prog_express,
            t2 - t1,
        )
    )

    best_count = best_prog.size

    ## break criteria
    if unchange_score == 200:
        break
    if pen_global_best < 0.05 and best_count < 2 ** 5 - 1:
        break
    # if pen_global_best < 0.01:
    #     break

    """
    ## parameter control
    if select_depth > 4 and best_count > 2**4 -1:
        select_depth = select_depth -2
        REG_STRENGTH = REG_STRENGTH +2
        print("updated select_depth: ", select_depth)
        print("updated REG_STRENGTH: ", REG_STRENGTH, "\n")



    if pen_global_best > 100 and select_depth < 20:
        select_depth = select_depth +2
        REG_STRENGTH = round(REG_STRENGTH -0.3, 1)
        print("updated select_depth: ", select_depth, "\n")
        print("updated REG_STRENGTH: ", REG_STRENGTH, "\n")
    """

    ## make it easier to concate
    if unchange_score == 100:
        # select_depth = 10
        # REG_STRENGTH = 2
        TOURNAMENT_SIZE = 10
        NEW_POP_PCT = 0
        print("set select_depth: %d, REG_STRENGTH: %d, TOURNAMENT_SIZE: %d\n"
              % (
                  select_depth,
                  REG_STRENGTH,
                  TOURNAMENT_SIZE,
              ))

    # if "x" not in prog_express and "y" not in prog_express or best_count > 2 ** 5 - 1:
    #     if best_prog in population:
    #         print("REMOVE best_prog", "\n")
    #         pen_global_best = float("inf")
    #         population.remove(best_prog)
    #         population.append(tree(col_name))
    #     NEW_POP_PCT = 0.2
    #     population = [
    #         get_offspring(population, fitness, TOURNAMENT_SIZE)
    #         for _ in range(round(POP_SIZE * (1.0 - NEW_POP_PCT)))]
    # else:
    NEW_POP_PCT = 0.05
    if unchange_score > 99:
        NEW_POP_PCT = 0
    population = [
        get_offspring(population, fitness, TOURNAMENT_SIZE)
        for _ in range(round(POP_SIZE * (1.0 - NEW_POP_PCT)) - 1)]
    population.append(best_prog)
    print("new gen")

    new_pop = [tree(col_name) for _ in range(round(POP_SIZE * NEW_POP_PCT))]
    population = population + new_pop

tf = time.time()
print("Best score: %f" % pen_global_best)
print("Best program: %s" % prog_express)
print("Total time: %d sec" % (tf - ts))

# 修改為你要傳送的訊息內容
m = "\n" + "Total time: %d sec" % (tf - ts)
message = str(prog_express) + "\n" + m
# 修改為你的權杖內容
token = 'CCgjmKSEGamkEj9JvhuIkFNYTrpPKHyCb1zdsYRjo86'

lineNotifyMessage(token, message)