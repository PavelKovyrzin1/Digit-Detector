import numpy as np

# Класс для вершины
class Node:
    def __init__(self,
                 factor_ind,
                 threshold,
                 left_node,
                 right_node):
        self.factor_ind = factor_ind
        self.threshold = threshold

        self.left_node = left_node
        self.right_node = right_node

    def is_leaf(self):
        return False


# Класс для листа
class Leaf:
    def __init__(self,
                 predicted_value):
        self.predicted_value = predicted_value

    def is_leaf(self):
        return True


# Предсказание для объекта x
def decision_tree_predict(cur_node, x):
    if cur_node.is_leaf():
        return cur_node.predicted_value

    if x[cur_node.factor_ind] >= cur_node.threshold:
        return decision_tree_predict(cur_node.right_node, x)
    return decision_tree_predict(cur_node.left_node, x)


# Проверка хаотичности данных помощью критерия Джини
def gini_index(X):
    columns = len(X)

    if columns == 0:
        return 1

    row_size = len(X[0])

    result = 0
    counter = dict()

    for row in X:
        if row[row_size - 1] in counter:
            counter[row[row_size - 1]] += 1
        else:
            counter[row[row_size - 1]] = 1

    for value in counter:
        p = counter[value] / columns
        result += p * (1 - p)

    return result


# Функция для разбиения данных в вершине по фактору factor_ind и значению factor_value
def data_split(X, factor_ind, factor_value):
    X_right = []
    X_left = []

    for row in X:
        if row[factor_ind] >= factor_value:
            X_right.append(row)
        else:
            X_left.append(row)

    X_left = np.array(X_left)
    X_right = np.array(X_right)
    return X_left, X_right


# Подюор оптимальных значений в вершине с помощью критерия Джини
def optimal_split_parameters_grid_search(X):
    result = [-1, -1]
    answer = -100
    columns = len(X)
    row_size = len(X[0])

    gini = gini_index(X)

    for j in range(row_size - 1):
        for i in range(columns):
            X_left, X_right = data_split(X, j, X[i][j])
            gini_left = gini_index(X_left)
            gini_right = gini_index(X_right)
            value = gini - gini_left * (len(X_left) / columns) - gini_right * (len(X_right) / columns)
            if value > answer:
                answer = value
                result = [j, X[i][j]]

    return result[0], result[1]

# Построение дерева принятие решений
def build_node(X, cur_depth=1, max_depth=None):
    columns = len(X)
    row_size = len(X[0])

    arr_x = X[:, row_size - 1]

    unique, counts = np.unique(arr_x, return_counts=True)
    ind = np.argmax(counts)
    val = unique[ind]

    arr_x = list(arr_x)
    arr_x.sort()

    index, value = optimal_split_parameters_grid_search(X)
    X_left, X_right = data_split(X, index, value)

    if arr_x[0] == arr_x[columns - 1] or len(X_left) == 0 or len(X_right) == 0 or cur_depth - 1 == max_depth:
        return Leaf(val)
    
    indexes.append(index)

    left_node = build_node(X_left, cur_depth + 1, max_depth=max_depth)
    right_node = build_node(X_right, cur_depth + 1, max_depth=max_depth)

    return Node(index, value, left_node, right_node)