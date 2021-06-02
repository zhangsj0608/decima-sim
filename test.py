import numpy as np
import tensorflow as tf
import copy
from spark_env.env import Environment
from spark_agent import SparkAgent
from heuristic_agent import DynamicPartitionAgent
from actor_agent import ActorAgent
from spark_env.canvas import *
from param import *
from utils import *
from fixed_tpch import query_idx_list, query_size_list


args.job_folder = './spark_env/tpch/'
args.alibaba_path = './spark_env/alibaba/batch_instance_3.csv'
args.saved_model = './models_3000ep/model_ep_3000'
args.query_type = 'fixed_tpch'
args.tpch_jobs_percentage = 0.5
args.result_folder = './results/'
args.exec_cap = 5
args.num_init_dags = 30  # init batch dag num
args.num_stream_dags = 0  # stream dag num
args.canvs_visualization = False

# max num of jobs 500
args.query_idx_list = query_idx_list
args.query_size_list = query_size_list

# create result folder
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)


def test_scheduler_with_recurrent_jobs(scheme='learn'):
    # args to re-enqueue finished jobs to the sys
    max_jobs_num = 100
    re_enqueue_prob = 0.7

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
    elif scheme == 'dynamic_partition':
        agent = DynamicPartitionAgent()
    elif scheme == 'spark_fifo':
        agent = SparkAgent(exec_cap=args.exec_cap)
    else:
        print('scheme ' + str(scheme) + ' not recognized')
        exit(1)

    # reset environment with seed
    env.seed(args.num_ep)  # fixed seed
    env.reset()

    # reset set of recurrent jobs  -- zsj
    selected_job_names = set()
    max_size_chosen_set = max_jobs_num - args.num_init_dags - args.num_stream_dags
    np_random = np.random
    np_random.seed(seed=max_jobs_num)

    # start experiment
    obs = env.observe()
    total_reward = 0
    done = False
    while not done:
        node, use_exec = agent.get_action(obs)
        obs, reward, done = env.step(node, use_exec)
        total_reward += reward

        # re-enqueue one job with a dedicated probability  -- zsj
        if len(selected_job_names) < max_size_chosen_set:
            selected_job = select_finished_job(env, selected_job_names, np_random)
            succeeded = add_copied_job_to_queue(env, selected_job, np_random, probability=re_enqueue_prob)
            if succeeded:
                selected_job_names.add(selected_job.name)

    if args.canvs_visualization:
        visualize_dag_time_save_pdf(env.finished_job_dags, env.executors, args.result_folder + 'visualization_exp_' +
                                    '_scheme_' + scheme + '.png', plot_type='app')
    else:
        visualize_executor_usage(env.finished_job_dags, args.result_folder + 'visualization_exp_' +
                                 '_scheme_' + scheme + '.png')

    jobs = env.all_jobs_ever_exist
    print('all jobs:', jobs)
    print('re_enqueued jobs: ', len(selected_job_names), selected_job_names)

    sorted_timeline = env.sorted_timeline
    sorted_timeline.sort(key=lambda x: x[0])
    for t, str_job in sorted_timeline:
        print('time {}s: {}'.format(t / 1000, str_job))


def select_finished_job(env, selected_job_names, np_random):
    all_jobs = env.all_jobs_ever_exist
    completed_jobs = [job for job in all_jobs if job.completed and job.name not in selected_job_names]  # each job is selected once
    if len(completed_jobs) == 0:
        return None

    index = np_random.randint(len(completed_jobs))
    selected_job = completed_jobs[index]

    return selected_job


def add_copied_job_to_queue(env, selected_job, np_random, probability):
    if selected_job is None:
        return False
    copied_job = copy.deepcopy(selected_job)
    copied_job.reset()
    # copied_job.name += '_cp'
    rand_seed = np_random.rand()

    if rand_seed > probability:
        current_t = env.wall_time.curr_time
        t = current_t + int(np_random.exponential(args.stream_interval))  # still the exponential
        env.timeline.push(t, copied_job)
        env.sorted_timeline.append((t, str(copied_job)))
        return True
    return False


def test_scheduler_slowdown(scheme='learn'):
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
    elif scheme == 'dynamic_partition':
        agent = DynamicPartitionAgent()
    elif scheme == 'spark_fifo':
        agent = SparkAgent(exec_cap=args.exec_cap)
    else:
        print('scheme ' + str(scheme) + ' not recognized')
        exit(1)

    # reset environment with seed
    env.seed(args.num_ep)  # fixed seed
    env.reset()

    # start experiment
    obs = env.observe()
    total_reward = 0
    done = False
    while not done:
        node, use_exec = agent.get_action(obs)
        obs, reward, done = env.step(node, use_exec)
        total_reward += reward

    jobs = env.all_jobs_ever_exist
    jct_all, makespan_all, slowdown_all = get_slowdown(jobs)

    visualize_executor_usage(env.finished_job_dags, args.result_folder + 'visualization_exp_' +
                             '_scheme_' + scheme + '.png')

    print('all jobs in sequence:', jobs)
    print('jct avg (s):', np.average(list(jct_all.values())), 'all:', jct_all)
    print('makespan max (s):', max(list(makespan_all.values())), 'avg', np.mean(list(makespan_all.values())), 'all', makespan_all)
    print('slowdown avg:', np.average(list(slowdown_all.values())), 'all:', slowdown_all)


def get_slowdown(completed_job_dags):
    slowdown_all_jobs = {}
    completion_time_all_jobs = {}
    mkspan_all_jobs = {}
    for job in completed_job_dags:
        submission_time = job.start_time
        start_time = np.nan
        finish_time = np.nan
        for node in job.nodes:
            for task in node.tasks:
                if np.isnan(start_time):
                    start_time = task.start_time
                elif start_time > task.start_time:
                    start_time = task.start_time

                if np.isnan(finish_time):
                    finish_time = task.finish_time
                elif finish_time < task.finish_time:
                    finish_time = task.finish_time
        slowdown = (finish_time - submission_time + 0.0) / (finish_time - start_time + 0.0)
        jct = finish_time - start_time
        mkspan = finish_time - submission_time
        slowdown_all_jobs[job] = slowdown
        completion_time_all_jobs[job] = jct / 1000
        mkspan_all_jobs[job] = mkspan / 1000
    return completion_time_all_jobs, mkspan_all_jobs, slowdown_all_jobs


def test_job_perturbation(job_id, scheme='learn', perturb='edge'):
    # Tensorflow seeding
    tf.set_random_seed(args.seed)  # fixed seed

    # set up environment
    env = Environment()
    if scheme == 'learn':
        sess = tf.Session()
        agent = ActorAgent(
            sess, args.node_input_dim, args.job_input_dim,
            args.hid_dims, args.output_dim, args.max_depth,
            range(1, args.exec_cap + 1))
    elif scheme == 'dynamic_partition':
        agent = DynamicPartitionAgent()
    elif scheme == 'spark_fifo':
        agent = SparkAgent(exec_cap=args.exec_cap)
    else:
        print('scheme ' + str(scheme) + ' not recognized')
        exit(1)

    # reset environment with seed
    env.seed(args.num_ep)  # fixed seed
    env.reset()

    init_jobs = env.job_dags
    chosen_job = init_jobs.to_list()[job_id]  # choose one job for job perturbation
    print('Chosen job:', chosen_job, 'num nodes:', chosen_job.num_nodes)

    if perturb == 'node':  # perturb one node from the graph
        for chosen_node in chosen_job.nodes:
            original_duration = chosen_node.tasks[-1].duration
            chosen_node.tasks[-1].duration = 0.0001  # halve the task duration of the node feature
            print('Perturbation: original', original_duration, 'current duration:', chosen_node.tasks[-1].duration)

    if perturb == 'edge':  # add edges to the graph
        adj_matrix = chosen_job.adj_mat
        for i in range(adj_matrix.shape[0]):
            for j in range(i, adj_matrix.shape[0]):
                adj_matrix[i][j] = 1
        print(chosen_job.adj_mat)

    # start experiment
    obs = env.observe()
    total_reward = 0
    done = False
    while not done:
        node, use_exec = agent.get_action(obs)
        obs, reward, done = env.step(node, use_exec)
        total_reward += reward

    jobs = env.all_jobs_ever_exist
    jct_all, makespan_all, slowdown_all = get_slowdown(jobs)
    print('jct', jct_all[chosen_job], 'makespan', makespan_all[chosen_job], 'slowdown', slowdown_all[chosen_job])
    print('avg jct', np.average(list(jct_all.values())), 'avg makespan', np.average(list(makespan_all.values())),
          'avg slowdown', np.average(list(slowdown_all.values())))


if __name__ == '__main__':
    test_job_perturbation(job_id=3, scheme='learn', perturb='node')

