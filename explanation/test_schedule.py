import tensorflow as tf
import numpy as np
import os

from spark_env.env import Environment
from spark_agent import SparkAgent
from explanation.heuristic_agent import DynamicPartitionAgent
from actor_agent import ActorAgent
from param import args
from fixed_tpch import query_idx_list, query_size_list
from explanation.critical_path import get_job_critical_path
from explanation.snf_agent import SmallestNodeFirstAgent
from spark_env.canvas import visualize_executor_usage
from explanation.perturbation import Perturbation


args.job_folder = '../spark_env/tpch/'
args.alibaba_path = '../spark_env/alibaba/batch_instance_3.csv'
args.saved_model = '../models_3000ep/model_ep_3000'
args.query_type = 'fixed_tpch'
args.tpch_jobs_percentage = 0.5
args.result_folder = './results/'
args.exec_cap = 5
args.num_init_dags = 30  # init batch dag num
args.num_stream_dags = 0  # stream dag num
args.canvs_visualization = False

# max num of jobs 500
args.query_idx_list = query_idx_list[120: ]
args.query_size_list = query_size_list[120: ]

# create result folder
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)


def get_critical_path_lengths(env):
    """get the run time and num tasks in the critical path of each job"""
    all_jobs = env.job_dags
    all_critical_paths = {a_job: get_job_critical_path(a_job) for a_job in all_jobs}
    all_critical_path_lengths = {}  # length of node_duration in cp

    for a_job, cp in all_critical_paths.items():
        remaining_tasks = [a_job.nodes[node_idx].num_tasks - a_job.nodes[node_idx].next_task_idx for node_idx in cp]
        node_durations = [(a_job.nodes[node_idx].num_tasks - a_job.nodes[node_idx].next_task_idx) *
                          a_job.nodes[node_idx].tasks[-1].get_duration() for node_idx in cp]
        cp_length = np.round(np.sum(node_durations) / 1000, 2)  # in seconds
        remaining_tasks_total = np.sum(remaining_tasks)
        all_critical_path_lengths[a_job] = (cp_length, remaining_tasks_total)
    return all_critical_path_lengths


def get_remaining_run_time(env):
    """get the run time and num tasks in each job"""
    all_jobs = env.job_dags
    remaining_time = {}
    for a_job in all_jobs:
        run_time = 0.0
        remaining_num_tasks = 0
        for node in a_job.nodes:
            run_time += (node.num_tasks - node.next_task_idx) * node.tasks[-1].get_duration()
            remaining_num_tasks += node.num_tasks - node.next_task_idx
        remaining_time[a_job] = (np.round(run_time / 1000, 2), remaining_num_tasks)
    return remaining_time


def get_smallest_ready_nodes(env):
    """get the run time, and num tasks in the smallest ready node of each job"""
    all_jobs = env.job_dags
    smallest_node_runtime = {}
    for a_job in all_jobs:
        frontiers = a_job.frontier_nodes
        if len(frontiers) == 0:
            smallest_node_runtime[a_job] = (0.0, 0.0)  # job has finished
        else:
            node_durations = [(node.num_tasks - node.next_task_idx) * node.tasks[-1].get_duration() for node
                              in frontiers]
            remaining_num_tasks = [node.num_tasks - node.next_task_idx for node in frontiers]
            smallest_node_runtime[a_job] = (np.round(min(node_durations) / 1000, 2), np.sum(remaining_num_tasks))
    return smallest_node_runtime


def test_scheduler(scheme='learn'):
    # tensorflow seeding
    tf.set_random_seed(args.seed)  # fixed seed

    # set up environment
    env = Environment()
    if scheme == 'learn':
        sess = tf.Session()
        agent = ActorAgent(
            sess, args.node_input_dim, args.job_input_dim,
            args.hid_dims, args.output_dim, args.max_depth,
            range(1, args.exec_cap + 1))
        short_name = 'learn'
    elif scheme == 'smallest_node_first':
        agent = SmallestNodeFirstAgent()
        short_name = 'snf'
    elif scheme == 'dynamic_partition':
        agent = DynamicPartitionAgent()
        short_name = 'dp'
    elif scheme == 'spark_fifo':
        agent = SparkAgent(exec_cap=args.exec_cap)
        short_name = 'fifo'
    else:
        print('scheme ' + str(scheme) + ' not recognized')
        exit(1)

    # reset environment with seed
    env.seed(args.num_ep)  # fixed seed
    env.reset()

    if isinstance(agent, SmallestNodeFirstAgent):
        agent.env = env

    # get the features for each job in each step
    all_cp_lengths = []
    all_remaining_runtime = []
    all_min_node_runtime = []

    # start experiment
    obs = env.observe()
    total_reward = 0
    done = False
    all_schedule_decisions = []

    step = 0  # for test -- zsj
    while not done:
        print('step', step)
        # get schedule action
        node, use_exec = agent.get_action(obs)
        if node is not None:
            # get features of each job in the schedule step
            cp_lengths = get_critical_path_lengths(env)  # runtime along the critical path of each job
            all_cp_lengths.append(cp_lengths)

            remaining_runtime = get_remaining_run_time(env)
            all_remaining_runtime.append(remaining_runtime)  # remaining runtime for each job

            min_node_runtime = get_smallest_ready_nodes(env)
            all_min_node_runtime.append(min_node_runtime)  # smallest frontier node's runtime in each job

            all_schedule_decisions.append((node.job_dag, use_exec, env.source_job,
                                           is_decided_by_algorithm(env.source_job, env.get_frontier_nodes())))

        obs, reward, done = env.step(node, use_exec)
        total_reward += reward
        step += 1

    visualize_executor_usage(job_dags=env.finished_job_dags,
                             file_path=args.result_folder + 'visual_{}_{}-jobs.png'.format(short_name, args.num_init_dags))
    # write schedule sequence and the critical path at each step
    with open(args.result_folder + 'schedule_{}_{}-jobs-5.txt'.format(short_name, args.num_init_dags), 'w') as file:
        for i in range(len(all_schedule_decisions)):
            file.write('step {}: job {}: Num_exec: {} Source job: {} Decided by algorithm: {}\n'.
                       format(i, all_schedule_decisions[i][0], all_schedule_decisions[i][1],
                              all_schedule_decisions[i][2], all_schedule_decisions[i][3]))
            for a_job in all_cp_lengths[i].keys():
                file.write(
                    str(a_job) + ',' + str(all_cp_lengths[i][a_job][0]) + ',' + str(all_cp_lengths[i][a_job][1]) + ','
                    + str(all_remaining_runtime[i][a_job][0]) + ',' + str(all_remaining_runtime[i][a_job][1]) + ','
                    + str(all_min_node_runtime[i][a_job][0]) + ',' + str(all_min_node_runtime[i][a_job][1]) + ','
                    + str(a_job == all_schedule_decisions[i][0]) + '\n')
            file.write('\n')

    all_jct = [job.completion_time / 1000 for job in env.finished_job_dags]
    print('jct: ', all_jct)
    print('average jct:', np.mean(all_jct))
    print('overall makespan:', max(all_jct))


def compare_scheduler(scheme_to_compare='smallest_node_first'):
    # tensorflow seeding
    tf.set_random_seed(args.seed)  # fixed seed

    # set up environment
    env = Environment()
    sess = tf.Session()
    drl_agent = ActorAgent(
        sess, args.node_input_dim, args.job_input_dim,
        args.hid_dims, args.output_dim, args.max_depth,
        range(1, args.exec_cap + 1))

    if scheme_to_compare == 'smallest_node_first':
        secondary_agent = SmallestNodeFirstAgent()
        short_name = 'snf'
    elif scheme_to_compare == 'dynamic_partition':
        secondary_agent = DynamicPartitionAgent()
        short_name = 'dp'
    else:
        secondary_agent = None

    # reset environment with seed
    env.seed(args.num_ep)  # fixed seed
    env.reset()

    if isinstance(secondary_agent, SmallestNodeFirstAgent):
        secondary_agent.env = env

    # start experiment
    obs = env.observe()
    done = False
    all_schedule_decisions = []

    chosen_node_idx_tensor = tf.placeholder(dtype=tf.int32, shape=[])
    node_gradients = tf.gradients(drl_agent.node_act_probs[0][chosen_node_idx_tensor], drl_agent.node_inputs)
    edge_gradients_tf = [tf.gradients(drl_agent.node_act_probs[0][chosen_node_idx_tensor], mat.values) for mat
                         in drl_agent.gcn.adj_mats]

    perturb = Perturbation(env, drl_agent, sess)

    while not done:
        # get schedule action
        node, use_exec = drl_agent.get_action(obs)

        # 1. check if it's the smallest node among all frontiers of the job
        if node is not None:
            candidate_nodes = [n for n in node.job_dag.nodes if n in env.get_frontier_nodes()]  # the same job's frontiers
        else:
            candidate_nodes = None

        # 2. compute the gradients on the inputs: node features and edges
        if node is not None:
            act_id = env.action_map.inverse_map[node]
            edge_gradients_np = perturb.evaluate_tensors(edge_gradients_tf, feed_dict={chosen_node_idx_tensor: act_id})
            important_edges = perturb.get_important_edges(edge_gradients_np,
                                                          mats=drl_agent.postman.get_msg_path(obs[0])[0],
                                                          max_num=5)
        else:
            important_edges = None

        # 3. record the decision at this step
        all_schedule_decisions.append(
            (node, use_exec, is_decided_by_algorithm(env.source_job, env.get_frontier_nodes()),
             is_smallest_node(node, candidate_nodes), candidate_nodes, important_edges))

        obs, reward, done = env.step(node, use_exec)  # proceed with drl scheduler's decision

    # write schedule sequence and the critical path at each step
    with open(args.result_folder + 'schedule_drl_{}-jobs-top-1.txt'.format(args.num_init_dags), 'w') as file:
        for i, (node, use_exec, decided_by_algorithm, choose_smallest_node, candidate_nodes, edge_importance) \
                in enumerate(all_schedule_decisions):
            file.write('step {}: \nNode {} use exec {} \ndecided by algorithm: {} is node_first_come_fist_serve: {}\nis smallest node of the job: {}\n'
                       'candidate nodes of the same jobs: {}\nedge importance: {}\n\n'
                .format(i, node, use_exec, decided_by_algorithm[0], decided_by_algorithm[1] == node, choose_smallest_node, candidate_nodes,
                        edge_importance))

    decision_made_by_algorithm = [item[2][0] for item in all_schedule_decisions]

    print('percentage decision by algorithm:', sum(decision_made_by_algorithm) / len(decision_made_by_algorithm))

    choose_smallest_node_by_algorithm = [item[3] for item in all_schedule_decisions if item[2][0]]
    first_come_first_serve_by_affinity = [item[2][1] == item[0] for item in all_schedule_decisions if not item[2][0]]

    print('smallest node by algorithm:', sum(choose_smallest_node_by_algorithm) / len(choose_smallest_node_by_algorithm))
    print('first_come_first_serve by affinity:', sum(first_come_first_serve_by_affinity) / len(first_come_first_serve_by_affinity))

    all_jct = [job.completion_time / 1000 for job in env.finished_job_dags]
    print('DRL jct: ', all_jct)
    print('DRL average jct:', np.mean(all_jct))


def flat_list(li):
    ans = []
    for item in li:
        if isinstance(item, list):
            ans.extend(flat_list(item))
        else:
            ans.append(item)
    return ans


def get_similarity(decisions_1, decisions_2):
    decisions_2_set = set(decisions_2)
    count = 0
    for decision in decisions_1:
        if decision in decisions_2_set:
            count += 1
    return (count + 0.0) / len(decisions_1)


def is_smallest_node(node, candidate_nodes):
    if node is None or candidate_nodes is None:
        return False

    remaining_time = [(node.idx, node.tasks[-1].duration * (node.num_tasks - node.next_task_idx)) for node
                      in candidate_nodes]

    smallest_idx = min(remaining_time, key=lambda x: x[1])[0]
    return node.idx == smallest_idx


def is_decided_by_algorithm(source_job, frontier_nodes):
    if source_job is None:
        return True, None
    if source_job is not None:
        # immediately scheduable nodes
        for node in source_job.frontier_nodes:
            if node in frontier_nodes:
                return False, node
        # schedulable node in the job
        for node in frontier_nodes:
            if node.job_dag == source_job:
                return False, node
        return True, None


if __name__ == '__main__':
    compare_scheduler('smallest_node_first')
    # test_scheduler('learn')

