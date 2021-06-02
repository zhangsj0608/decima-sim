import os
from param import *
from utils import *
from spark_env.task import *
from spark_env.node import *
from spark_env.job_dag import *
from spark_env.alibaba_data_loader import CSVReader, generate_jobs_from_configs


def load_job(file_path, query_size, query_idx, wall_time, np_random):
    query_path = file_path + query_size + '/'
    
    adj_mat = np.load(
        query_path + 'adj_mat_' + str(query_idx) + '.npy', allow_pickle=True)
    task_durations = np.load(
        query_path + 'task_duration_' + str(query_idx) + '.npy', allow_pickle=True).item()
    
    assert adj_mat.shape[0] == adj_mat.shape[1]
    assert adj_mat.shape[0] == len(task_durations)

    num_nodes = adj_mat.shape[0]
    nodes = []
    for n in range(num_nodes):
        task_duration = task_durations[n]
        e = next(iter(task_duration['first_wave']))

        num_tasks = len(task_duration['first_wave'][e]) + \
                    len(task_duration['rest_wave'][e])

        # remove fresh duration from first wave duration
        # drag nearest neighbor first wave duration to empty spots
        pre_process_task_duration(task_duration)
        rough_duration = np.mean(
            [i for l in task_duration['first_wave'].values() for i in l] + \
            [i for l in task_duration['rest_wave'].values() for i in l] + \
            [i for l in task_duration['fresh_durations'].values() for i in l])

        # generate tasks in a node
        tasks = []
        for j in range(num_tasks):
            task = Task(j, rough_duration, wall_time)
            tasks.append(task)

        # generate a node
        node = Node(n, tasks, task_duration, wall_time, np_random)
        nodes.append(node)

    # parent and child node info
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                nodes[i].child_nodes.append(nodes[j])
                nodes[j].parent_nodes.append(nodes[i])

    # initialize descendant nodes
    for node in nodes:
        if len(node.parent_nodes) == 0:  # root
            node.descendant_nodes = recursive_find_descendant(node)

    # generate DAG
    job_dag = JobDAG(nodes, adj_mat,
        args.query_type + '-' + query_size + '-' + str(query_idx))

    return job_dag


def pre_process_task_duration(task_duration):
    # remove fresh durations from first wave
    clean_first_wave = {}
    for e in task_duration['first_wave']:
        clean_first_wave[e] = []
        fresh_durations = SetWithCount()
        # O(1) access
        for d in task_duration['fresh_durations'][e]:
            fresh_durations.add(d)
        for d in task_duration['first_wave'][e]:
            if d not in fresh_durations:
                clean_first_wave[e].append(d)
            else:
                # prevent duplicated fresh duration blocking first wave
                fresh_durations.remove(d)

    # fill in nearest neighour first wave
    last_first_wave = []
    for e in sorted(clean_first_wave.keys()):
        if len(clean_first_wave[e]) == 0:
            clean_first_wave[e] = last_first_wave
        last_first_wave = clean_first_wave[e]

    # swap the first wave with fresh durations removed
    task_duration['first_wave'] = clean_first_wave


def recursive_find_descendant(node):
    if len(node.descendant_nodes) > 0:  # already visited
        return node.descendant_nodes
    else:
        node.descendant_nodes = [node]
        for child_node in node.child_nodes:  # terminate on leaves automatically
            child_descendant_nodes = recursive_find_descendant(child_node)
            for dn in child_descendant_nodes:
                if dn not in node.descendant_nodes:  # remove dual path duplicates
                    node.descendant_nodes.append(dn)
        return node.descendant_nodes


def generate_alibaba_jobs(np_random, timeline, wall_time):
    """generate alibaba jobs in exponential time interval"""

    csv_reader = CSVReader(max_lines=10000)
    csv_reader.load_data_from_raw_csv(filename=args.alibaba_path)
    csv_reader._filter_jobs_with_multi_task()
    csv_reader._filter_jobs_with_correct_structure()

    init_job_configs = csv_reader.random_generate(number=args.num_init_dags, np_random=np_random, batch_mode=True)
    init_job_dags = generate_jobs_from_configs(init_job_configs, wall_time, np_random)

    job_dags = OrderedSet()
    t = 0
    for job_dag in init_job_dags:
        job_dag.start_time = t
        job_dag.arrived = True
        job_dags.add(job_dag)

    stream_job_configs = csv_reader.random_generate(number=args.num_stream_dags, np_random=np_random, batch_mode=True)
    stream_job_dags = generate_jobs_from_configs(stream_job_configs, wall_time, np_random)
    for job_dag in stream_job_dags:
        t += int(np_random.exponential(args.stream_interval))
        job_dag.start_time = t
        timeline.push(t, job_dag)

    return job_dags


def generate_tpch_jobs(np_random, timeline, wall_time):

    job_dags = OrderedSet()
    t = 0

    for _ in range(args.num_init_dags):
        # generate query
        query_idx = str(np_random.randint(args.tpch_num) + 1)
        query_size = args.tpch_size[np_random.randint(len(args.tpch_size))]
        # generate job
        job_dag = load_job(
            args.job_folder, query_size, query_idx, wall_time, np_random)
        # job already arrived, put in job_dags
        job_dag.start_time = t
        job_dag.arrived = True
        job_dags.add(job_dag)

    for _ in range(args.num_stream_dags):
        # poisson process
        t += int(np_random.exponential(args.stream_interval))
        # uniform distribution
        query_size = args.tpch_size[np_random.randint(len(args.tpch_size))]
        query_idx = str(np_random.randint(args.tpch_num) + 1)
        # generate job
        job_dag = load_job(
            args.job_folder, query_size, query_idx, wall_time, np_random)
        # push into timeline
        job_dag.start_time = t
        timeline.push(t, job_dag)

    return job_dags


def generate_fixed_tpch_jobs(np_random, timeline, wall_time, query_idx_list, query_size_list):

    print('[INFO <job generator>]: fixed tpch jobs are generated')
    job_dags = OrderedSet()
    t = 0
    idx = 0  # batch jobs
    for _ in range(args.num_init_dags):
        # generate query
        query_idx = query_idx_list[idx]
        query_size = query_size_list[idx]
        # generate job
        job_dag = load_job(
            args.job_folder, query_size, query_idx, wall_time, np_random)
        # job already arrived, put in job_dags
        job_dag.start_time = t
        job_dag.arrived = True
        job_dags.add(job_dag)
        idx += 1

    # stream jobs
    for _ in range(args.num_stream_dags):
        # poisson process
        t += int(np_random.exponential(args.stream_interval))  # 指数分布
        query_size = query_size_list[idx]
        query_idx = query_idx_list[idx]
        # generate job
        job_dag = load_job(
            args.job_folder, query_size, query_idx, wall_time, np_random)
        # push into timeline
        job_dag.start_time = t
        timeline.push(t, job_dag)
        idx += 1

    return job_dags


def generate_mixed_tpch_alibaba_jobs(np_random, timeline, wall_time, tpch_jobs_percent, query_idx_list,
                                     query_size_list):
    print('[INFO <job generator>]: fixed tpch jobs & alibaba jobs are generated')
    csv_reader = CSVReader(max_lines=10000)
    csv_reader.load_data_from_raw_csv(filename=args.alibaba_path)
    csv_reader._filter_jobs_with_multi_task()
    csv_reader._filter_jobs_with_correct_structure()

    job_dags = OrderedSet()
    t = 0
    idx = 0
    for _ in range(args.num_init_dags):
        # generate query
        query_idx = query_idx_list[idx]
        query_size = query_size_list[idx]
        # generate job
        job_dag = load_job(
            args.job_folder, query_size, query_idx, wall_time, np_random)
        # job already arrived, put in job_dags
        job_dag.start_time = t
        job_dag.arrived = True
        job_dags.add(job_dag)
        idx += 1

    # tpch jobs (stream)
    total_tpch_jobs = int(args.num_stream_dags * tpch_jobs_percent)
    for i in range(total_tpch_jobs):
        # poisson process
        t += int(np_random.exponential(args.stream_interval))  # 指数分布

        query_size = query_size_list[idx]
        query_idx = query_idx_list[idx]
        # generate job
        job_dag = load_job(
            args.job_folder, query_size, query_idx, wall_time, np_random)
        # push into timeline
        job_dag.start_time = t
        timeline.push(t, job_dag)
        idx += 1

    # alibaba jobs
    alibaba_job_configs = csv_reader.random_generate(number=args.num_stream_dags - total_tpch_jobs,
                                                     np_random=np_random,
                                                     batch_mode=True)
    alibaba_job_dags = generate_jobs_from_configs(alibaba_job_configs, wall_time, np_random)
    for job_dag in alibaba_job_dags:
        t += int(np_random.exponential(args.stream_interval))
        job_dag.start_time = t
        timeline.push(t, job_dag)

    return job_dags


def generate_jobs(np_random, timeline, wall_time):
    sorted_timeline = None
    if args.query_type == 'tpch':
        job_dags = generate_tpch_jobs(np_random, timeline, wall_time)

    elif args.query_type == 'alibaba':
        job_dags = generate_alibaba_jobs(np_random, timeline, wall_time)

    elif args.query_type == 'fixed_tpch':
        job_dags = generate_fixed_tpch_jobs(np_random, timeline, wall_time,
                                                             query_idx_list=args.query_idx_list,
                                                             query_size_list=args.query_size_list)

    elif args.query_type == 'mixed':
        job_dags = generate_mixed_tpch_alibaba_jobs(np_random, timeline, wall_time,
                                                    tpch_jobs_percent=args.tpch_jobs_percentage,
                                                    query_idx_list=args.query_idx_list,
                                                    query_size_list=args.query_size_list)

    else:
        print('Invalid query type ' + args.query_type)
        exit(1)

    return job_dags
