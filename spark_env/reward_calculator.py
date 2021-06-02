import numpy as np
from param import *


class RewardCalculator(object):
    def __init__(self):
        self.job_dags = set()
        self.prev_time = 0

    def get_reward(self, job_dags, curr_time):
        reward = 0

        # add new job into the store of jobs
        for job_dag in job_dags:
            self.job_dags.add(job_dag)

        # now for all jobs (may have completed)
        # compute the elapsed time
        if args.learn_obj == 'mean':
            for job_dag in list(self.job_dags):
                reward -= (min(
                    job_dag.completion_time,
                    curr_time) - max(
                    job_dag.start_time,
                    self.prev_time)) / \
                    args.reward_scale

                # if the job is done, remove it from the list
                if job_dag.completed:
                    self.job_dags.remove(job_dag)

        elif args.learn_obj == 'makespan':
            reward -= (curr_time - self.prev_time) / \
                args.reward_scale

        elif args.learn_obj == 'makespan_new':
            reward = 0
            remain_duration = 0
            num_remain_nodes = 0.000001
            for job_dag in self.job_dags:
                available_nodes = [node for node in job_dag.nodes if not node.no_more_tasks]
                if len(available_nodes) == 0:
                    continue

                num_remain_nodes += len(available_nodes)
                remain_duration += sum([node.tasks[-1].get_duration() * (node.num_tasks - node.next_task_idx)
                                        for node in available_nodes])
            avg_remain_duration = remain_duration  # / + num_remain_nodes
            reward -= avg_remain_duration / args.reward_scale
        else:
            print('Unkown learning objective')
            exit(1)

        self.prev_time = curr_time

        return reward

    def reset(self):
        self.job_dags.clear()
        self.prev_time = 0
