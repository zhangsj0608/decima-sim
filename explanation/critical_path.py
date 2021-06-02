import numpy as np


class Edge:
    def __init__(self, id, start, end, weight):
        self.id = id
        self.start = start
        self.end = end
        self.weight = weight
    
    def print_Edge(self):
        print('edge_id : {}, edge_start : {}, edge_end : {}, edge_weight : {}'.format(self.id, self.start, self.end, self.weight))


#G:邻接表
def topsort(G):
    in_degrees = dict((u, 0) for u in G)
    for u in G:
        for v in  G[u]:
            in_degrees[v] += 1
                                    # 每一个节点的入度
    Q = [u for u in G if in_degrees[u] == 0]
                                    # 入度为 0 的节点
    S = []
    while Q:
        u = Q.pop()
                                    # 默认从最后一个移除
        S.append(u)
        for v in G[u]:
            in_degrees[v] -= 1
            if in_degrees[v] == 0:
                Q.append(v)
    S = [int(i) for i in S]
    return S


#邻接矩阵转邻接表
def to_table(graph):
    G = {}
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if graph[i][j] != -1:
                if i in G:
                    G[i].append(j)
                else:
                    l = []
                    l.append(j)
                    G[i] = l
    G[len(G)] = []
    return G


#事件的最早执行时间
def Ve_list(graph, toplist):
    Ve = [0 for i in range(len(toplist))]
    for i in toplist:
        for cur in range(len(graph[i])):
            if graph[i][cur] >= 0:
                if Ve[cur] < Ve[i] + graph[i][cur]:
                    Ve[cur] = Ve[i] + graph[i][cur]
    return Ve


#事件的最晚执行时间
def Vl_list(graph_V, toplist, last):
    Vl = [last for i in range(len(toplist))]
    stack = []
    stack.append(toplist[-1])
    while len(stack) > 0:
        i = stack.pop()
        for cur in range(len(graph_V[i])):
            if graph_V[i][cur] >= 0:
                stack.append(cur)
                if Vl[cur] > Vl[i] - graph_V[i][cur]:
                    Vl[cur] = Vl[i] - graph_V[i][cur]
    return Vl


#活动的最早、最晚开始时间
def e_l_list(edge, Vl, Ve, last):
    early_list = [0 for i in range(len(edge))]
    last_list = [last for i in range(len(edge))]
    path = []
    for e in edge:
        early_list[e.id] = Ve[e.start]
        last_list[e.id] = Vl[e.end] - e.weight
    for cur in range(len(edge)):
        if abs(early_list[cur] - last_list[cur]) <= 0.001:
            path.append(edge[cur])
    # print(early_list, last_list)
    return path


#得到图的所有边
def get_edge(graph):
    edge, cur = [], 0
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i][j] != -1:
                e = Edge(cur, i, j, graph[i][j])
                edge.append(e)
                cur += 1
    return edge


#得到关键路径上的点
def edge_node(edge, start, end):
    node = []
    node.append(start)
    for cur in edge:
        if cur.start == start:
            break
    while cur.end != end:
        for i in edge:
            if cur.end == i.start:
                cur = i
                break
        node.append(cur.start)
    node.append(end)
    return node


def get_critical_path(graph):
    toplist = topsort(to_table(graph))
    Ve = Ve_list(graph, toplist)
    last = max(Ve)
    graph_V = list(map(list, (zip(*graph))))
    Vl = Vl_list(graph_V, toplist, last)
    edge = e_l_list(get_edge(graph), Vl, Ve, last)
    path = edge_node(edge, toplist[0], toplist[-1])
    return path


def get_job_critical_path(job):
    """get the node ids that on the critical path"""
    adj_mat = job.adj_mat
    nodes = job.nodes

    mat = adj_mat.copy()

    has_children = np.squeeze(np.any(mat, axis=0))
    new_row = [0 if not flag else -1 for flag in has_children]
    new_col = [[-1] for _ in range(len(nodes) + 1)]

    for i in range(len(nodes)):
        total_length = nodes[i].get_node_duration()
        for j in range(len(nodes)):
            if mat[i][j] == 0:
                mat[i, j] = -1
            else:
                mat[i, j] = total_length

    mat = np.vstack([new_row, mat])
    mat = np.hstack([new_col, mat])
    critical_path = get_critical_path(mat)
    critical_path = [n_id - 1 for n_id in critical_path[1:]]

    return critical_path


if __name__ == '__main__':
    graph = [[-1, 0, 0, -1, 0, -1, -1],
    [-1, -1, -1, 4, -1, 4, -1],
    [-1, -1, -1, 48, -1, -1, -1,],
    [-1, -1, -1, -1, -1, 100, -1],
    [-1, -1, -1, -1, -1, 36, -1],
    [-1, -1, -1, -1, -1, -1, 18],
    [-1, -1, -1, -1, -1, -1, -1]]
    path = get_critical_path(graph)
    print(path)
