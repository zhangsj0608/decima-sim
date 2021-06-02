import numpy as np
import tensorflow as tf
from msg_passing_path import get_unfinished_nodes_summ_mat


class Perturbation(object):
    def __init__(self, env, agent, session):
        self.env = env
        self.agent = agent
        self.sess = session

    def evaluate_tensors(self, tensor_list, feed_dict=None):
        """
        execute tf and evaluate the tensors
        :param tensor_list: tensor list to be executed
        :param node_inputs: feature matrix, which is a numpy array
        :return:
        """
        obs = self.env.observe()
        agent = self.agent
        _, job_inputs, job_dags, source_job, num_source_exec, frontier_nodes, executor_limits, exec_commit, \
        moving_executors, exec_map, action_map = agent.translate_state(obs)

        node_inputs = agent.translate_state(obs)[0]

        _, gcn_masks, dag_summ_backward_map, running_dags_mat, job_dags_changed = agent.postman.get_msg_path(
            job_dags)

        gcn_mats = agent.postman.get_msg_path(job_dags)[0]

        node_valid_mask, job_valid_mask = \
            agent.get_valid_masks(job_dags, frontier_nodes, source_job, num_source_exec, exec_map, action_map)

        summ_mats = get_unfinished_nodes_summ_mat(job_dags)

        if feed_dict is None:
            feed_dict = {}
        for i, d in zip([agent.node_inputs] +
                        [agent.job_inputs] + [agent.node_valid_mask] +
                        [agent.job_valid_mask] + agent.gcn.adj_mats + agent.gcn.masks +
                        agent.gsn.summ_mats + [agent.dag_summ_backward_map],
                        [node_inputs] + [job_inputs] + [node_valid_mask] +
                        [job_valid_mask] + gcn_mats + gcn_masks +
                        [summ_mats, running_dags_mat] + [dag_summ_backward_map]):
            feed_dict[i] = d
        output_np = self.sess.run(tensor_list, feed_dict=feed_dict)
        return output_np

    def get_important_edges(self, mat_gradients_np, mats, max_num=1):

        edge_gradient_li = []
        for mat, gradients in zip(mats, mat_gradients_np):
            for edge, gradient in zip(mat.indices, gradients[0]):
                edge_gradient_li.append(((self.get_node(int(edge[0, 0])), self.get_node(int(edge[0, 1]))),
                                         gradient))

        edge_gradient_li.sort(key=lambda x: x[1], reverse=True)
        return edge_gradient_li[: max_num]

    def get_node(self, node_id):
        return self.env.action_map[node_id]


