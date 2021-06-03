import numpy as np

# TODO(zsj): Define C4.5 decision tree for init job_level schedule approximation, including training, testing, explanation.
class TreeNode(object):
    N_ID = 0
    def __ini__(attr, vals):
        self.n_id = N_ID
        self.attr = attr  # selected attribute name
        self.attr_vals = vals  # values for the selected attributes
        self.gain_ratio = None
        self.child_nodes = []
        self.is_leaf = False
        self.major_class = None

        N_ID += 1


class DecisionTree(object):

    def __init__(attributes_with_types, gain_lambda):
        self.attributes = list(attributes_with_types.keys())
        self.attribute_types = attributes_with_types

        self.gain_lambda = gain_lambda
        self.root = TreeNode(None, None)
        self.nodes = []

    def build_tree(instances):
        # build tree recursively over instances.
        tree_node_and_instances = {self.head: instances} # node: data instances
        tree_nodes = [self.head]  # used as queue
        for cur_node in tree_nodes:
            node, grouped_instances = create_node(cur_node, tree_node_and_instances[cur_node], self.attributes)

        if node is 

        pass

    def create_node(cur_node, instances, candidate_attributes):
        # gain ratio after split in regarding the chosen attribute. chosen_attr_values is the attrbute values. 
        max_gain_ratio, chosen_attr, chosen_attr_values, grouped_instances = 0, None, None, None
        for attr in candidate_attributes:
            if self.attribute_types[attr] == 'discrete':
                compute_entropy_gain_ratio = compute_entropy_gain_ratio_discrete_attribute
            else:
                compute_entropy_gain_ratio = compute_entropy_gain_ratio_continues_attribute

            entropy_gain_ratio, grouped_instances_ = compute_entropy_gain_ratio(instances, attr)
            if entropy_gain_ratio > max_gain_ratio:
                max_gain_ratio, chosen_attr, chosen_attr_values, grouped_instances = entropy_gain_ratio, attr, list(grouped_instances_.keys()), grouped_instances_

        if max_gain_ratio >= self.gain_lambda:  # assure the gain_ratio is greater than the bar
            new_node = TreeNode(chosen_attr, chosen_attr_values)
            new_node.gain_ratio = max_gain_ratio
            cur_node.child_nodes.append(new_node)
            return new_node, grouped_instances
        else:
            return None, None

    def compute_entropy_gain_ratio_discrete_attribute(instances, attribute):
        label_id = -1  # the id in an instance for label
        attribute_id = None  # the id in an instance for selected attribute
        for i, attr in enumerate(self.attributes):
            if attr == attribute:
                attribute_id = i
                break
        
        gain_ratio, grouped_instances = get_entropy_gain_ratio(instances, attribute)
        return gain_ratio, grouped_instances

    def compute_entropy_gain_ratio_continues_attribute(instances, attribute):
        attribute_id = 0
        for i, attr in enumerate(self.attributes):
            if attr == attribute:
                attribute_id = i
        label_id = -1
        attr_val_and_labels = [(float(instance[attribute_id]), instance[label_id]) for instance in instances]
        attr_val_and_labels.sort(key=lambda x: x[0])
        cur_label = None
        split_boundraies = []
        for val, label in attr_val_and_labels:
            if cur_label is None or label != cur_label:
                split_boundraies.append(val)
                cur_label = label

        max_gain_ratio, selected_grouped_instances = None, None
        for boundary in split_boundraies:
            gain_ratio, grouped_instances = get_entropy_gain_ratio(instances, attribute, boundary)
            if max_gain_ratio is None or gain_ratio > max_gain_ratio:
                max_gain_ratio, selected_grouped_instances = gain_ratio, grouped_instances
        return max_gain_ratio, selected_grouped_instances

    def get_entropy_gain_ratio(instances, attribute, val_boundary=None):
        statistic_original = {'positive': 0, 'negative': 0}
        statistic_after_split = {}
        statistic_intrinsic = {}
        grouped_instances = {}
        for instance in instances:
            if val_boundary is None:  # discrete attribute
                attribute_val = instance[attribute_id]
            else:  # continues attribute
                if instance[attribute_id] < val_boundary:
                    attribute_val = 'less_than_val_boundary_%.2f' % val_boundary
                else:
                    attribute_val = 'greater_than_val_boundary_%.2f' % val_boundary
            if attribute_val not in statistic_after_split:
                statistic_after_split[attribute_val] = {'positive': 0, 'negative': 0}
                statistic_intrinsic[attribute_val] = 0
                grouped_instances[attribute_val] = []

            if instance[label_id] == 'True':
                label = 'postive'
            else:
                label = 'negative'

            statistic_after_split[attribute_val][label] += 1
            statistic_original[label] += 1
            statistic_intrinsic[attribute_val] += 1
            grouped_instances[attribute_val].append(instance)  # instances is grouped by attribute values

        
        # original = {'p': 1577, 'n': 25811}
        # locality = {'p': 1388, 'n': 177}
        # nonlocality = {'p': 189, 'n': 25634}
        summ_ori = statistic_original['positive'] + statistic_original['negative']
        summ_after_split = {}
        for attr_val in statistic_after_split.keys():
            summ_after_split[attr_val] = statistic_after_split[attr_val]['positive'] + statistic_after_split[attr_val]['negative']

        # summ_loc = locality['p'] + locality['n']
        # summ_non_loc = nonlocality['p'] + nonlocality['n']
        gain = get_entropy(statistic_original)  # the entropy before split

        # en_loc = entropy(locality)
        # en_non_loc = entropy(nonlocality)
        for attr_val in statistic_after_split.keys():
            gain -= summ_after_split[attr_val] / summ_ori * get_entropy(statistic_after_split[attr_val])

        # gain = en_ori - (summ_loc / summ_ori * en_loc + summ_non_loc / summ_ori * en_non_loc)
        # iv = - summ_loc / summ_ori * np.log2(summ_loc / summ_ori) - summ_non_loc / summ_ori * np.log2(summ_non_loc / summ_ori)
        iv = get_entropy(summ_after_split)
        gain_ratio = gain / iv
        return gain_ratio, grouped_instances
        
    

def get_entropy(info):
    summ = 0
    for key in info.keys():
        summ += info[key]
    # summ = info['positive'] + info['negative']
    entropy = 0
    for key in info.keys():
        entropy += - info[key] / summ * np.log2(info[key] / summ)
    # entropy = - info['positive'] / summ * np.log2(info['positive'] / summ) - info['negative'] / summ * np.log2(info['negative'] / summ)
    return entropy

