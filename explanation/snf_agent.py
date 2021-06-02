import numpy as np

import heapq
from param import *
from agent import Agent
from spark_env.job_dag import JobDAG
from spark_env.node import Node


class SmallestNodeFirstAgent(Agent):
    # dynamically partition the cluster resource
    # scheduling complexity: O(num_nodes * num_executors)
    def __init__(self):
        Agent.__init__(self)
        self.env = None

    def get_action(self, obs, **kwargs):
        if 'selected_nodes_current_stage' in kwargs:  # remove duplicate selection if in a same stage
            selected_nodes_current_stage = kwargs['selected_nodes_current_stage']
        else:
            selected_nodes_current_stage = None

        if 'num_smallest_nodes' in kwargs:  # find the smallest nodes (> 1) in test mode
            assert 'test' in kwargs and kwargs['test'] is True
            num_smallest_nodes = kwargs['num_smallest_nodes']  # only used for test
        else:
            num_smallest_nodes = 1  # for real schedule, the num_smallest_nodes should be always 1

        decision_made_by_algorithm = False
        # parse observation
        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map = obs

        # sort out the exec_map
        exec_map = {}  # current executor_num allocated to each job
        for job_dag in job_dags:
            exec_map[job_dag] = len(job_dag.executors)
        # count in moving executors
        for node in moving_executors.moving_executors.values():
            exec_map[node.job_dag] += 1
        # count in executor commit
        for s in exec_commit.commit:
            if isinstance(s, JobDAG):
                j = s
            elif isinstance(s, Node):
                j = s.job_dag
            elif s is None:
                j = None
            else:
                print('source', s, 'unknown')
                exit(1)
            for n in exec_commit.commit[s]:
                if n is not None and n.job_dag != j:
                    exec_map[n.job_dag] += exec_commit.commit[s][n]

        # first assign executor to the same job (source) -- 优先原来的job
        if source_job is not None:
            # immediately scheduable nodes
            for node in source_job.frontier_nodes:
                if node in frontier_nodes:
                    return node, num_source_exec, decision_made_by_algorithm
            # schedulable node in the job
            for node in frontier_nodes:
                if node.job_dag == source_job:
                    return node, num_source_exec, decision_made_by_algorithm

        # job_dag.frontier_nodes -> node ready to schedule for run  - zsj
        # env.frontier_nodes -> node that can be schedule later (not saturated) - zsj

        # the source job is finished or does not exist
        decision_made_by_algorithm = True
        remain_time_list = []
        frontier_node_list = []
        if len(frontier_nodes) != 0:
            for node in frontier_nodes:
                if selected_nodes_current_stage is not None and node in selected_nodes_current_stage:
                    continue
                frontier_node_list.append(node)
        else:
            for job_dag in job_dags:
                for node in job_dag.frontier_nodes:
                    if selected_nodes_current_stage is not None and node in selected_nodes_current_stage:
                        continue
                    if node not in self.env.node_selected:
                        frontier_node_list.append(node)

        for node in frontier_node_list:
            remaining_run_time = 0.0
            for task_id in range(node.next_task_idx, node.num_tasks):
                remaining_run_time += node.tasks[task_id].duration  # rough duration.
            remain_time_list.append(remaining_run_time)  # for each node (schedulable node)

        next_node = None
        n_smallest = None
        if len(remain_time_list) != 0:
            min_node_idx = np.argmin(remain_time_list)
            next_node = frontier_node_list[min_node_idx]

            if num_smallest_nodes > 1:  # find multiple smallest nodes in test mode
                n_smallest_tuple = heapq.nsmallest(num_smallest_nodes, zip(remain_time_list, frontier_node_list),
                                             key=lambda x: x[0])
                n_smallest = [y for x, y in n_smallest_tuple]

        # node is selected, compute limit
        if next_node is not None:
            use_exec = min(
                next_node.num_tasks - next_node.next_task_idx - \
                exec_commit.node_commit[next_node] - \
                moving_executors.count(next_node), num_source_exec, 2)  # 2 兼顾了fairness,每个node最多一个executor
            if n_smallest is not None:
                return n_smallest, use_exec, decision_made_by_algorithm   # return multiple smallest nodes

            return next_node, use_exec, decision_made_by_algorithm

        # there is more executors than tasks in the system
        return None, num_source_exec, decision_made_by_algorithm
