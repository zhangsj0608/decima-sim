import tensorflow as tf
import numpy as np

from typing import List, Dict
from spark_env.env import Environment
from spark_env.job_dag import JobDAG
from spark_env.node import Node
from utils import OrderedSet
from actor_agent import ActorAgent
from param import args
from fixed_tpch import query_idx_list, query_size_list
from spark_env.canvas import visualize_executor_usage


def perturbation_with_node_split(job_dag: JobDAG, node_idx: int, splits: List[int]) -> JobDAG:
    target_node = job_dag.nodes[node_idx]
    old_adj_mat = job_dag.adj_mat

    # split the old target_node to some new nodes, each containing part of tasks
    new_nodes = []
    task_idx = 0
    for i, num in enumerate(splits):  # split the target_node to len(splits) new nodes
        if num != -1:
            tasks = target_node.tasks[task_idx: task_idx + num]
        else:
            tasks = target_node.tasks[task_idx:]
        new_node = Node(idx=node_idx + i, tasks=tasks, task_duration=target_node.task_duration,
                        wall_time=target_node.wall_time, np_random=target_node.np_random)
        new_nodes.append(new_node)
        task_idx += num

    for node in job_dag.nodes[node_idx + 1:]:
        node.idx += len(new_nodes) - 1

    # synthesise the new nodes with other old nodes, and create the new adj mat for all nodes
    all_nodes = job_dag.nodes[: node_idx] + new_nodes + job_dag.nodes[node_idx + 1:]
    new_adj_mat = np.zeros((len(all_nodes), len(all_nodes)))
    for i in range(old_adj_mat.shape[0]):
        for j in range(old_adj_mat.shape[1]):
            if old_adj_mat[i, j] == 1:
                if i < node_idx:
                    if j < node_idx:
                        new_adj_mat[i, j] = 1
                    elif j > node_idx:
                        new_adj_mat[i, j + len(new_nodes) - 1] = 1
                    else:
                        new_adj_mat[i, node_idx: (node_idx + len(new_nodes))] = 1
                elif i == node_idx:
                    if j < node_idx:
                        new_adj_mat[node_idx: node_idx + len(new_nodes), j] = 1
                    elif j > node_idx:
                        new_adj_mat[node_idx: node_idx + len(new_nodes), j + len(new_nodes) - 1] = 1
                else:
                    if j < node_idx:
                        new_adj_mat[i + len(new_nodes) - 1, j] = 1
                    elif j > node_idx:
                        new_adj_mat[i + len(new_nodes) - 1, j + len(new_nodes) - 1] = 1
                    else:
                        new_adj_mat[i + len(new_nodes) - 1, node_idx: node_idx + len(new_nodes)] = 1

    # amend the job_dag to contain the new_adj_mat and all_nodes
    job_dag.nodes = all_nodes
    job_dag.adj_mat = new_adj_mat
    job_dag.num_nodes = len(all_nodes)

    for node in job_dag.nodes:
        node.child_nodes = []
        node.parent_nodes = []
        node.job_dag = job_dag  # assign job to nodes

    for i in range(job_dag.num_nodes):
        for j in range(job_dag.num_nodes):
            if job_dag.adj_mat[i, j] == 1:
                job_dag.nodes[i].child_nodes.append(job_dag.nodes[j])
                job_dag.nodes[j].parent_nodes.append(job_dag.nodes[i])

    job_dag.frontier_nodes = OrderedSet()
    for node in job_dag.nodes:
        if node.is_schedulable():
            job_dag.frontier_nodes.add(node)
    return job_dag


def schedule_with_perturbation(pert_opt: Dict = None):
    # tensorflow seeding
    tf.set_random_seed(args.seed)  # fixed seed
    sess = tf.Session()
    agent = ActorAgent(
        sess, args.node_input_dim, args.job_input_dim,
        args.hid_dims, args.output_dim, args.max_depth,
        range(1, args.exec_cap + 1))

    # set up environment
    env = Environment()

    # perturbation applied here
    if pert_opt is not None:
        target_job_idx = pert_opt['job_idx']
        node_idx = pert_opt['node_idx']
        splits = pert_opt['splits']
        # changed_job = perturbation_with_node_split(target_job, node_idx, splits)
        pert_func = lambda job_list: perturbation_with_node_split(job_list[target_job_idx], node_idx, splits)
    else:
        pert_func = None

    # reset environment with seed
    env.seed(args.num_ep)  # fixed seed
    env.reset(pert_func=pert_func)

    # start experiment
    obs = env.observe()
    done = False
    while not done:
        # get schedule action
        node, use_exec = agent.get_action(obs)
        obs, reward, done = env.step(node, use_exec)

    visualize_executor_usage(job_dags=env.finished_job_dags,
                             file_path=args.result_folder + 'schedule_perturb.png')
    completion_time = [job_dag.completion_time / 1000 for job_dag in env.finished_job_dags]
    return completion_time


def set_args():
    args.job_folder = '../spark_env/tpch/'
    args.alibaba_path = '../spark_env/alibaba/batch_instance_3.csv'
    args.saved_model = '../models_3000ep/model_ep_3000'
    args.query_type = 'fixed_tpch'
    args.tpch_jobs_percentage = 0.5
    args.result_folder = './results/'
    args.exec_cap = 5
    args.num_init_dags = 10  # init batch dag num
    args.num_stream_dags = 0  # stream dag num
    args.canvs_visualization = False

    # max num of jobs 500
    args.query_idx_list = query_idx_list
    args.query_size_list = query_size_list


if __name__ == '__main__':
    set_args()
    perturb_job_idx = 9  # 0 <= idx < args.num_init_jobs
    perturb_node_idx = 0
    node_splits = [1, -1]
    pert_opt = {'job_idx': perturb_job_idx, 'node_idx': perturb_node_idx, 'splits': node_splits}
    # pert_opt = None
    completion_time = schedule_with_perturbation(pert_opt=pert_opt)
    print('all completion time', completion_time)
    print('target_job jct:', completion_time[perturb_job_idx])
