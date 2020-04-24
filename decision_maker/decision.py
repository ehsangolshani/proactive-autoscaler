from decision_maker.cluster_state import ClusterState
from decision_maker.topsis import TOPSIS
from pprint import pprint


class DecisionMaker:
    def __init__(self, cluster_state: ClusterState, topsis: TOPSIS):
        self.cluster_state = cluster_state
        self.topsis = topsis

    def decide(self,
               total_future_required_cpu_units,
               total_future_required_memory_units,
               total_future_required_gpu_units):
        cpu_scale_out_required_available_proportion = \
            total_future_required_cpu_units / self.cluster_state.calculate_total_available_cpu(scale_out=True)
        cpu_no_scale_required_available_proportion = \
            total_future_required_cpu_units / self.cluster_state.calculate_total_available_cpu()
        cpu_scale_down_required_available_proportion = \
            total_future_required_cpu_units / self.cluster_state.calculate_total_available_cpu(
                scale_down=True) if self.cluster_state.node_number > 1 else 0

        memory_scale_out_required_available_proportion = \
            total_future_required_memory_units / self.cluster_state.calculate_total_available_memory(scale_out=True)
        memory_no_scale_required_available_proportion = \
            total_future_required_memory_units / self.cluster_state.calculate_total_available_memory()
        memory_scale_down_required_available_proportion = \
            total_future_required_memory_units / self.cluster_state.calculate_total_available_memory(
                scale_down=True) if self.cluster_state.node_number > 1 else 0

        gpu_scale_out_required_available_proportion = \
            total_future_required_gpu_units / self.cluster_state.calculate_total_available_gpu(scale_out=True)
        gpu_no_scale_required_available_proportion = \
            total_future_required_gpu_units / self.cluster_state.calculate_total_available_gpu()
        gpu_scale_down_required_available_proportion = \
            total_future_required_gpu_units / self.cluster_state.calculate_total_available_gpu(
                scale_down=True) if self.cluster_state.node_number > 1 else 0

        # calculate criteria

        cpu_scale_out_violation = self.cluster_state.sla.calculate_cpu_violation(
            value=cpu_scale_out_required_available_proportion)
        cpu_no_scale_violation = self.cluster_state.sla.calculate_cpu_violation(
            value=cpu_no_scale_required_available_proportion)
        cpu_scale_down_violation = self.cluster_state.sla.calculate_cpu_violation(
            value=cpu_scale_down_required_available_proportion)

        memory_scale_out_violation = self.cluster_state.sla.calculate_memory_violation(
            value=memory_scale_out_required_available_proportion)
        memory_no_scale_violation = self.cluster_state.sla.calculate_memory_violation(
            value=memory_no_scale_required_available_proportion)
        memory_scale_down_violation = self.cluster_state.sla.calculate_memory_violation(
            value=memory_scale_down_required_available_proportion)

        gpu_scale_out_violation = self.cluster_state.sla.calculate_gpu_violation(
            value=gpu_scale_out_required_available_proportion)
        gpu_no_scale_violation = self.cluster_state.sla.calculate_gpu_violation(
            value=gpu_no_scale_required_available_proportion)
        gpu_scale_down_violation = self.cluster_state.sla.calculate_gpu_violation(
            value=gpu_scale_down_required_available_proportion)

        cpu_scale_out_under_utilization = self.cluster_state.sla.calculate_cpu_under_utilization(
            value=cpu_scale_out_required_available_proportion)
        cpu_no_scale_under_utilization = self.cluster_state.sla.calculate_cpu_under_utilization(
            value=cpu_no_scale_required_available_proportion)
        cpu_scale_down_under_utilization = self.cluster_state.sla.calculate_cpu_under_utilization(
            value=cpu_scale_down_required_available_proportion)

        memory_scale_out_under_utilization = self.cluster_state.sla.calculate_memory_under_utilization(
            value=memory_scale_out_required_available_proportion)
        memory_no_scale_under_utilization = self.cluster_state.sla.calculate_memory_under_utilization(
            value=memory_no_scale_required_available_proportion)
        memory_scale_down_under_utilization = self.cluster_state.sla.calculate_memory_under_utilization(
            value=memory_scale_down_required_available_proportion)

        gpu_scale_out_under_utilization = self.cluster_state.sla.calculate_gpu_under_utilization(
            value=gpu_scale_out_required_available_proportion)
        gpu_no_scale_under_utilization = self.cluster_state.sla.calculate_gpu_under_utilization(
            value=gpu_no_scale_required_available_proportion)
        gpu_scale_down_under_utilization = self.cluster_state.sla.calculate_gpu_under_utilization(
            value=gpu_scale_down_required_available_proportion)

        scale_out_cost = self.cluster_state.calculate_total_resource_cost(scale_out=True)
        no_scale_cost = self.cluster_state.calculate_total_resource_cost()
        scale_down_cost = self.cluster_state.calculate_total_resource_cost(scale_down=True)

        item_criteria_matrix = [
            [cpu_scale_out_violation,
             memory_scale_out_violation,
             gpu_scale_out_violation,
             cpu_scale_out_under_utilization,
             memory_scale_out_under_utilization,
             gpu_scale_out_under_utilization,
             scale_out_cost],
            [cpu_no_scale_violation,
             memory_no_scale_violation,
             gpu_no_scale_violation,
             cpu_no_scale_under_utilization,
             memory_no_scale_under_utilization,
             gpu_no_scale_under_utilization,
             no_scale_cost
             ],
        ]

        if self.cluster_state.node_number > 1:
            item_criteria_matrix.append(
                [cpu_scale_down_violation,
                 memory_scale_down_violation,
                 gpu_scale_down_violation,
                 cpu_scale_down_under_utilization,
                 memory_scale_down_under_utilization,
                 gpu_scale_down_under_utilization,
                 scale_down_cost]
            )

        # print('decision-matrix -->  ')
        # pprint(item_criteria_matrix)

        decision_ranks = self.topsis.rank(matrix=item_criteria_matrix)
        return decision_ranks
