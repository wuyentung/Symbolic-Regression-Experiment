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

class GlobalParameter:
    def __init__(self):
        self.col_names = ["x1", 'x2', 'y1']
        self.pop_size = 500
        self.tournament_size = 3
        self.df = pd.DataFrame(data=[[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["x1", 'x2', 'y1'])

GLOBAL = GlobalParameter()
def set_global_DATA(df):
    '''

    :param df:
    :type df: pd.DataFrame
    :return:
    '''
    setattr(GLOBAL, "df", df)
    col = df.columns.to_list()
    setattr(GLOBAL, "col_names", col)
    # print("GLOBAL.df: ", GLOBAL.df)
    # print("GLOBAL.col_names: ", GLOBAL.col_names)

def set_global_POP_SIZE(POP_SIZE):
    setattr(GLOBAL, "pop_size", POP_SIZE)

def set_global_TOURNAMENT_SIZE(TOURNAMENT_SIZE):
    setattr(GLOBAL, "tournament_size", TOURNAMENT_SIZE)


#%%
## operation setting
def safe_div(a, b):
    return a / b if b else a
def safe_pow(a, b):
    return np.abs(a) ** np.abs(b)
OPERATIONS = (
    {"func": np.add, "arg_count": 2, "format_str": "({} + {})", "check": "+"},
    {"func": np.subtract, "arg_count": 2, "format_str": "({} - {})", "check": "-"},
    {"func": np.multiply, "arg_count": 2, "format_str": "({} * {})", "check": "*"},
    {"func": np.true_divide, "arg_count": 2, "format_str": "({} / {})", "check": "/"},
    {"func": np.float_power, "arg_count": 2, "format_str": "({} ** {})", "check": "**"},
    {"func": np.negative, "arg_count": 1, "format_str": "-({})", "check": "minus"},
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
NUMBER_INT = np.arange(10).tolist()
NUMBER_DECI = []
NUMBERS = []
for i in range(10):
    # NUMBERS.append(NUMBER_INT[i])
    if i == 0:
        continue
    NUMBER_DECI.append(NUMBER_INT[i] / 10)
NUMBERS = NUMBER_INT + NUMBER_DECI
NUMBERS_WITHOUT0 = NUMBERS[1:-1]
NUMBER_DECI0 = [NUMBER_INT[0]] + NUMBER_DECI
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
def evaluate(root, df=GLOBAL.df):
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
            # print("check df")
            # print(df)
            # print([root.value] * df.shape[0])
            return [root.value] * df.shape[0]
        # print(df[root.value].to_list())
        return df[root.value].to_list()
    nodecheck(root, root.left)
    if 2 == root.arg_count:
        nodecheck(root, root.right)
        return root.value["func"](evaluate(root.left, df), evaluate(root.right, df))
    return root.value["func"](evaluate(root.left, df))

#%%
def simplify_constant(root):
    """ calculate value of root which is constant
    :type root: Node
    :rtype: Node
    """
    value = str(eval(root.program_express))
    return value

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
    if root.is_constant:
        # print("is_constant")
        express = root.value["format_str"].format(_build_tree_string(root.left), _build_tree_string(root.right))
        return round(eval(express), 4)
        if not root.simplified:
            # print("not root.simplified")
            root.simplified = True
            return "{}".format(simplify_constant(root=root))
    if root.is_leaf():
        return "{}".format(root.value)
    if 2 == root.arg_count:
        return root.value["format_str"].format(_build_tree_string(root.left), _build_tree_string(root.right))
    return root.value["format_str"].format(_build_tree_string(root.left))



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

def _get_leaves(root, leaf_type="var", col_name=GLOBAL.col_names):
    '''get leaves which is not Numbers

    :param leaf_type: type of leaves to get, default "var", can choose "number" or "all"
    :type leaf_type: str
    :param col_name:
    :type col_name: list
    :param root: tree root for the expression tree
    :type root: Node
    :return: Node which belongs to variables
    :rtype: [Node]
    '''
    var_leaves = []
    num_leaves = []
    leaves = []
    stack = [root]

    while stack:
        node = stack.pop()
        if node:
            if node.is_leaf():
                leaves.append(node)
                if isinstance(node.value, Number):
                    num_leaves.append(node)
                else:
                    var_leaves.append(node)
            stack.append(node.right)
            stack.append(node.left)
    if "var" == leaf_type:
        return var_leaves
    elif "number" == leaf_type:
        return num_leaves
    else:
        return leaves
#%%
def _get_nodes(root, col_name=GLOBAL.col_names):
    '''get leaves which is not Numbers

    :param leaf_type: type of leaves to get, default "var", can choose "number" or "all"
    :type leaf_type: str
    :param col_name:
    :type col_name: list
    :param root: tree root for the expression tree
    :type root: Node
    :return: Node which belongs to variables
    :rtype: [Node]
    '''
    nodes = []
    stack = [root]

    while stack:
        node = stack.pop()
        if node:
            nodes.append(node)
            stack.append(node.right)
            stack.append(node.left)
    return nodes
#%%
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
    def __init__(self, value, parent, leaf=False, left=None, right=None, is_constant=False, version="c-d"):
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
        self.var_leaves = None
        self.num_leaves = None
        self.leaves = None
        self.is_constant = is_constant
        self.simplified = False
        self.version = version
        self.children = None

    
    def set_parent(self, parent):
        """[summary]

        Args:
            parent ([Node]): [description]
        """
        self.parent = parent

    def is_leaf(self):
        if self.right == self.left is None:
            return True
        return False

    def program_print(self):
        """print program

        """
        print(self.program_express)
        pass

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

        # if self.is_constant:
        #     if not self.simplified:
        #         self.simplified = True
        #         self._program = simplify_constant(self)

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

            <BLANKLINE>
                1
               /
              2
             /
            3
            <BLANKLINE>
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

            <BLANKLINE>
              1____
             /     \\
            2       3
                   /
                  4
                 /
                5
            <BLANKLINE>
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
        """
        properties = _get_tree_properties(self)

        return properties

    @property
    def get_var_leaves(self):
        '''
        :rtype: list
        '''
        if self.var_leaves is None:
            self.var_leaves = _get_leaves(self)
        return self.var_leaves

    @property
    def get_num_leaves(self):
        '''
        :rtype: list
        '''
        if self.num_leaves is None:
            self.num_leaves = _get_leaves(root=self, leaf_type="num")
        return self.num_leaves

    @property
    def get_all_leaves(self):
        '''
        :rtype: list
        '''
        if self.leaves is None:
            self.leaves = _get_leaves(root=self, leaf_type="all")
        return self.leaves

    @property
    def get_children(self):
        '''
        :rtype: list
        '''
        if self.children is None:
            self.children = _get_nodes(root=self)
        return self.children

#%%
class Var_leaf:
    def __init__(self, node, input_flag):
        '''

        :param node:
        :type node: Node
        :type input_flag: bool
        '''
        self.node = node
        if input_flag:
            self.input_flag = True
            self.output_flag = False
        else:
            self.input_flag = False
            self.output_flag = True
        self.mul_parents = []
        self.exp_parents = []
        self.minus_parents = []
        self.exp_constant_positive = False
        self.positive = False
        
class Internal:
    def __init__(self, node):
        '''
        :type node: Node
        '''
        self.node = node

def production_properties(root, col_name=GLOBAL.col_names):
    '''

    :param root:
    :type root: Node
    :param col_name:
    :type col_name: list
    :return:
    '''
    # C-D positive A property, True if fit this property
    def positive_A(minus_parents):
        if len(minus_parents) // 2:
            return False
        return True

    # C-D exp's constant property, True if fit this property
    def par_constant(node):
        positive = False
        if 0 == len(node.right.get_var_leaves):
            # since it's leaves are all numbers, it's value is constant, so we calculate first
            value = evaluate(node.right)
            if value[0] > 0:
                positive = True
            node.right = Node(value=value[0], parent=node, leaf=True)

        return positive

    # get var_leaves first
    temp_storage = root.get_var_leaves
    # tagging
    var_leaves = []
    for leaf in temp_storage:
        input_flag = False
        if "x" in leaf.value:
            input_flag = True
        var_leaves.append(Var_leaf(node=leaf, input_flag=input_flag))
    # set up leaves properties
    for leaf in var_leaves:
        parent = leaf.node.parent
        child = leaf.node
        parents = []
        # TODO: confirm constraints
        while parent:
            if parent.value["check"] == "*":
                leaf.mul_parents.append(parent)
            # prepare for alpha/beta constant
            elif parent.value["check"] == "**":
                if parent.left == child:
                    leaf.exp_constant_positive = par_constant(parent)
                leaf.exp_parents.append(parent)


            # prepare for positive_A
            elif parent.value["check"] == "minus":
                leaf.minus_parents.append(parent)
            elif parent.value["check"] == "-":
                if parent.right == child:
                    leaf.minus_parents.append(parent)
            if parent.parent is None:
                final_parent = parent
            child = parent
            parent = parent.parent
        leaf.positive = positive_A(leaf.minus_parents)
    # for each leaf record each parent who is "*", "**", or "-"

#%%
class Coe:
    """can only be > 0 or ≥ 0, ≥ 0 if can0 is True
    """
    def __init__(self, can0: bool) -> None:
        self.can0 = can0
        pass

#%%
def generate_coe(is_alpha=False, is_can0=False, is_cannot0=False, is_int_n=False):
    if is_alpha:
        slack = random.choice(NUMBER_DECI0)
        alpha = random.choice(NUMBER_DECI)
        while slack + alpha >= 1:
            if 0.9 == slack:
                slack = random.choice(NUMBER_DECI0)
            alpha = random.choice(NUMBER_DECI)
        beta = round(1 - alpha - slack, 1)
        return alpha, beta

    if is_can0:
        a = round(random.random() * 10, 4)
        pct0 = 0.05
        if pct0 > random.random():
            a = 0
        return a

    if is_cannot0:
        a = round(random.random() * 10, 4)
        while 0 == a:
            a = round(random.random() * 10, 4)
        return a

    if is_int_n:
        return random.randint(-3, 2)
#%%
def cb_production_tree(parent=None, variables=GLOBAL.col_names):
    """[use Cobb-Douglas production function to produce tree]
    form:
               *
          /         \             
        A             *
                   /     \        
                **        **  
               /  \      /   \  
            x1   alpha  x2    beta

        s.t. A, alpha, beta > 0
            alpha + beta ≤ 1    -> alpha + beta + slack = 1
                                   slack ≥ 0

    form of A:
              *
            /   \    
           a     b

        s.t. a, b is number not 0
 
    Args:
        variables ([str], optional): [description]. Defaults to GLOBAL.col_names.

    Returns:
        Node: root of structure-defined tree, should be left subtree of function's root which value == "+"
    """
    # "*" is on OPERATIONS[2]
    # "**" is on OPERATIONS[4]

    # operator setting
    root = Node(value=OPERATIONS[2], parent=parent)
    root.right = Node(value=OPERATIONS[2], parent=root)
    root.right.left = Node(value=OPERATIONS[4], parent=root.right)
    root.right.right = Node(value=OPERATIONS[4], parent=root.right)
    
    # A setting, use tree(coe=Coe)
    A_coe = Coe(can0=False)
    root.left = sci_coe(parent=root, can0=A_coe.can0)

    # alpha, beta setting
    # slack = random.choice(NUMBER_DECI0)
    # alpha = random.choice(NUMBER_DECI)
    # while slack + alpha >= 1:
    #     if 0.9 == slack:
    #         slack = random.choice(NUMBER_DECI0)
    #     alpha = random.choice(NUMBER_DECI)
    # beta = round(1 - alpha - slack, 1)
    alpha, beta = generate_coe(is_alpha=True)
    # print(alpha, beta)

    root.right.left.right = Node(value=alpha, parent=root.right.left, leaf=True)
    root.right.right.right = Node(value=beta, parent=root.right.right, leaf=True)
    
    # x1, x2 setting
    x = []
    for var in variables:
        if "x" in var:
            x.append(var)
    # print(x)
    ## x1
    root.right.left.left = Node(value=x[0], parent=root.right.left, leaf=True)
    ## x2
    root.right.right.left = Node(value=x[1], parent=root.right.right, leaf=True)
    # root.right.left.program_print()
    # root.right.right.program_print()
    
    return root

#%%
def var_tree(variable, parent=None, linear=True):
    """Function: coe * var ^ coe, s.t. coe > 0, for x: ≥ 0
    form:
               *
          /         \             
       coe1           **
                   /     \        
                 var      coe2  
    linear form:
               *
          /         \             
       coe1           var

    Args:
        variable ([type]): [description]

    Returns:
        Node: [description]
    """
    # "*" is on OPERATIONS[2]
    # "**" is on OPERATIONS[4]
    
    # setting coe1, coe2
    coe1 = Coe(can0=False)
    coe2 = Coe(can0=True)
    if "x" in variable:
        coe1 = Coe(can0=True)

    root = Node(value=OPERATIONS[2], parent=parent)
    root.left = sci_coe(parent=root, can0=coe1.can0)
    if linear:
        root.right = Node(value=variable, parent=root.right, leaf=True)
        return root

    root.right = Node(value=OPERATIONS[4], parent=root)
    root.right.left = Node(value=variable, parent=root.right, leaf=True)
    root.right.right = sci_coe(parent=root.right, can0=coe2.can0)

    return root

#%%
def sci_coe(parent=None, can0=False):
    ''' scientific express for coe, which a* 10^n, 1<= a < 10, n is int
    :type parent: Node
    :rtype: Node
    '''

    # setting a, n
    # print("do sci_coe")
    if can0:
        a = generate_coe(is_can0=True)
    else:
        a = generate_coe(is_cannot0=True)
        
    n = generate_coe(is_int_n=True)

    # build tree of coe
    root = Node(value=OPERATIONS[2], parent=parent, is_constant=True)
    root.left = Node(value=a, parent=root, leaf=True)
    root.right = Node(value=OPERATIONS[4], parent=root)
    root.right.left = Node(value=10, parent=root.right, leaf=True)
    root.right.right = Node(value=n, parent=root.right, leaf=True)

    return root



#%%
def tree(variables=GLOBAL.col_names, height=4, depth=0, cobb=True, parent=None, is_coe=False):
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
                # coe
                if variables is None:
                    operant = random.choice(NUMBERS)
                else:
                    operant = random.choice(random.choice((NUMBERS, variables)))
                if getattr(root, attr) is None:
                    setattr(root, attr, Node(value=operant, parent=root, leaf=True))
                else:
                    _insert_random_leaf_values(getattr(root, attr))
        pass

    generating_height = max(int(height - depth), 0)

    if generating_height > 1:
        # print(type(height))
        operations: list
        operations = _generate_random_internode_values(generating_height - 1)
        inter_leaf_count = _generate_random_internode_count(generating_height - 1)
        root = Node(operations.pop(0), parent=parent)
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
        root = Node(value=random.choice(random.choice((NUMBERS, variables))), parent=None, leaf=True)

    # coe
    if isinstance(is_coe, Coe):
        # root.program_print()
        value = evaluate(root=root)
        if is_coe.can0:
            if value < 0:
                root = tree(variables=variables, height=height, cobb=False, parent=parent, is_coe=is_coe)
        if value <= 0:
            root = tree(variables=variables, height=height, cobb=False, parent=parent, is_coe=is_coe)
        value = evaluate(root=root)
        # root.program_print()
        # print(value)

    # cobb-douglas funtion
    if cobb:
        # oprator structure
        root = Node(value=random.choice([OPERATIONS[0], OPERATIONS[1]]), parent=parent)
        root.left = Node(value=OPERATIONS[1], parent=root)
        root.left.left = Node(value=OPERATIONS[0], parent=root.left)
        root.left.left.left = Node(value=OPERATIONS[0], parent=root.left.left)

        # else
        ## b_coe
        b_coe = Coe(can0=True)
        root.right = sci_coe(parent=root, can0=b_coe.can0)
        ## variables
        y = []
        x = []
        for var in variables:
            if "x" in var:
                x.append(var)
            else:
                y.append(var)
        root.left.right = var_tree(variable=y[0], parent=root.left)
        root.left.left.right = var_tree(variable=x[0], parent=root.left.left)
        root.left.left.left.right = var_tree(variable=x[1], parent=root.left.left.left)
        root.left.left.left.left = cb_production_tree(parent=root.left.left.left)

        # cd_parent = parent
        # cd_root = Node(value=OPERATIONS[0], parent=cd_parent)
        # cd_root.left = cb_production_tree(variables=variables, parent=cd_root)

    return root

#%%
## mutation
def do_mutate(root, col_name=GLOBAL.col_names, MUTATE_PCT=0.1, version=2):
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
    elif version > 1 and version < 2:
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
    else:
        # according to tree structure, there'er 4 kind of coes, 10 coes total

        # [A > 0] *2
        # [a >= 0] *3
        # [int n] *4
        coe_nodes = [offspring.left.right.left.left, 
        offspring.left.left.left.left.left.left, 
        offspring.left.left.left.right.left.left, 
        offspring.left.left.right.left.left, 
        offspring.right.left, 
        offspring.left.left.left.right.left.right.right, 
        offspring.left.left.right.left.right.right, 
        offspring.left.right.left.right.right, 
        offspring.right.right.right]

        # [alpha, beta] *1
        alpha_beta = [offspring.left.left.left.left.right.left.right, offspring.left.left.left.left.right.right.right]

        # mutation
        for i in range(9):
            # print(i)
            if random.random() > MUTATE_PCT:
                continue
            if i < 2:
                # [A > 0] *2
                coe_nodes[i].value = generate_coe(is_cannot0=True)
            elif i < 5:
                # [a >= 0] *3
                coe_nodes[i].value = generate_coe(is_can0=True)
            else:
                # [int n] *4
                coe_nodes[i].value = generate_coe(is_int_n=True)
        if random.random() < MUTATE_PCT:
            # [alpha, beta] *1
            alpha, beta = generate_coe(is_alpha=True)
            alpha_beta[0].value = alpha
            alpha_beta[1].value = beta
        if random.random() < MUTATE_PCT:
            # [root] *1
            offspring.value = random.choice([OPERATIONS[0], OPERATIONS[1]])
        offspring._program = None

    return offspring



#%%
## crossover
def do_xover(selected1, selected2, version=2):
    '''

    :param selected1:
    :type selected1: Node
    :param selected2:
    :type selected2: Node
    :return:
    :rtype: Node
    '''
    offspring1 = deepcopy(selected1)
    offspring2 = deepcopy(selected2)
    if 1 == version:
        # print(offspring.program_print())
        # xover_parent1 = select_random_node(offspring, None, 0)
        # print(xover_parent1.program_print())
        # print("\np2\n")
        # print(selected2.program_print())
        xover_parent1, p1_level = offspring1.rand_internal()
        xover_parent2, p2_level = selected2.rand_internal()

        setattr(xover_parent1, _randChildAttr(xover_parent1), xover_parent2.rand_child())
        # if 2 == xover_parent1.arg_count:
        #     attr1 = random.choice((LEFT, RIGHT))
        # else:
        #     attr1 = LEFT
        # setattr(xover_parent1, attr1, xover_parent2.rand_child())
        return offspring1
    elif version >1 and version < 2:
        xover_parent1, p1_level = offspring1.rand_internal()
        xover_parent2, p2_level = offspring2.rand_internal()
        x_point1_LR = _randChildAttr(xover_parent1)
        x_point2_LR = _randChildAttr(xover_parent2)
        x_point1 = deepcopy(getattr(xover_parent1, x_point1_LR))
        x_point2 = deepcopy(getattr(xover_parent2, x_point2_LR))
        setattr(xover_parent1, x_point1_LR, x_point2)
        setattr(xover_parent2, x_point2_LR, x_point1)
        return offspring1, offspring2
    else:
        # according to tree structure, there'er 4 kind of coes
        # [A > 0] *2
        # [a >= 0] *3
        # [int n] *4
        # [alpha, beta] *1
        offsprings = [offspring1, offspring2]
        # selecteds = [selected1, selected2]
        # print("before:")
        # offspring1.program_print()
        # print()
        # offspring2.program_print()
        # print()
        # x_points = [None, None]
        rand = random.random()
        # rand = 0.95
        if rand < 0.2:
            # [A > 0] *2
            options = ["y1", "cobb_A"]
            
            def a_cannot0(i, option=random.choice(options)):
                if "y1" in option:
                    xover_point = offsprings[i].left.right.left.left
                else:
                    xover_point = offsprings[i].left.left.left.left.left.left
                return xover_point

            # set xover_point1
            xover_point1 = a_cannot0(0)
            # set xover_point2
            xover_point2 = a_cannot0(1)
            
            
        elif rand >=0.2 and rand < 0.5:
            # [a >= 0] *3
            options = ["x1", "x2", "b"]

            def a_can0(i, option=random.choice(options)):
                if "x1" in option:
                    xover_point = offsprings[i].left.left.left.right.left.left
                elif "x2" in option:
                    xover_point = offsprings[i].left.left.right.left.left
                else:
                    xover_point = offsprings[i].right.left
                return xover_point
            
            # set xover_point1
            xover_point1 = a_can0(0)
            # set xover_parent2
            xover_point2 = a_can0(1)

        elif rand >=0.5 and rand < 0.9:
            # [int n] *4
            options = ["x1", "x2", "y1", "b"]

            def int_n(i, option=random.choice(options)):
                if "x1" in option:
                    xover_point = offsprings[i].left.left.left.right.left.right.right
                elif "x2" in option:
                    xover_point = offsprings[i].left.left.right.left.right.right
                elif "y1" in option:
                    xover_point = offsprings[i].left.right.left.right.right
                else:
                    xover_point = offsprings[i].right.right.right
                return xover_point
            
            # set xover_point1
            xover_point1 = int_n(0)
            # set xover_parent2
            xover_point2 = int_n(1)
        else:
            # [alpha, beta] *1
            options = ["x1", "x2", "y1", "b"]

            def alpha(i):
                xover_point = offsprings[i].left.left.left.left.right.left.right
                return xover_point
            
            def beta(i):
                xover_point = offsprings[i].left.left.left.left.right.right.right
                return xover_point
            # set xover_alpha1
            xover_alpha1 = alpha(0)
            # set xover_alpha2
            xover_alpha2 = alpha(1)

            # set xover_beta1
            xover_beta1 = beta(0)
            # set xover_beta2
            xover_beta2 = beta(1)

            # xover alpha
            alpha1_value = deepcopy(xover_alpha1.value)
            alpha2_value = deepcopy(xover_alpha2.value)
            xover_alpha1.value = alpha2_value
            xover_alpha2.value = alpha1_value
            
            # xover beta
            beta1_value = deepcopy(xover_beta1.value)
            beta2_value = deepcopy(xover_beta2.value)
            xover_beta1.value = beta2_value
            xover_beta2.value = beta1_value

            offspring1._program = None
            offspring2._program = None
            # print("\nwe select:")
            # print(alpha1_value)
            # print(alpha2_value)

            # print("\nand:")
            # print(beta1_value)
            # print(beta2_value)

            # print("\nafter:")
            # offspring1.program_print()
            # print()
            # offspring2.program_print()

            return offspring1, offspring2
        
        # for i in range(2):
        #     nodes = offsprings[i].get_children
        #     for node in nodes:
        #         node.simplified = False
        point1_value = deepcopy(xover_point1.value)
        point2_value = deepcopy(xover_point2.value)
        # print("\nwe select:")
        # print(point1_value)
        # print(point2_value)

        xover_point1.value = point2_value
        offspring1._program = None
        xover_point2.value = point1_value
        offspring2._program = None
        # print("\nafter:")
        # offspring1.program_print()
        # print()
        # offspring2.program_print()
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
def get_random_root(population, fitness=False, POP_SIZE=GLOBAL.pop_size, TOURNAMENT_SIZE=3):
    if fitness:
        # randomly select population members for the tournament
        tournament_members = [
            random.randint(0, POP_SIZE - 1) for _ in range(TOURNAMENT_SIZE)]
        # print("check: ")
        # print(len(fitness))
        # print(fitness)
        # print(len(population))
        # print(population)
        # select tournament member with best fitness
        member_fitness = [(fitness[i], population[i]) for i in tournament_members]
        # print("pick: ")
        # min(member_fitness, key=lambda x: x[0])[1].program_print()

        return min(member_fitness, key=lambda x: x[0])[1]


    # OR randomly select population members
    return random.choice(population)

## get_offspring
## TODO: XOVER_PCT into fun.
def get_offspring(population, fitness, POP_SIZE=GLOBAL.pop_size, col_name=GLOBAL.col_names, TOURNAMENT_SIZE=3, XOVER_PCT=0.7, version=1.1):
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
        offsprings[0], offsprings[1] = do_xover(parent1, parent2)
        for i in range(2):
            offsprings[i] = do_mutate(offsprings[i], col_name)
        return offsprings[0], offsprings[1]

    elif 1.2 == version:
        parent1 = get_random_root(population)
        parent2 = get_random_root(population)
        offsprings[0], offsprings[1] = do_xover(parent1, parent2, version=version)
        for i in range(2):
            offsprings[i] = do_mutate(offsprings[i], col_name, version=version)
        return offsprings[0], offsprings[1]

def selection(population, offsprings, fitness_pop, fitness_off, POP_SIZE=GLOBAL.pop_size, method="WHEEL"):
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
    temp2= _get_leaves(tt, col_name=0)

    print(df)
    tt.program_print()
    print(temp2)
    #%%
    print(temp2[1])




