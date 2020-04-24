from decision_maker.cluster_state import ClusterState
from decision_maker.decision import DecisionMaker
from decision_maker.sla import SLA
from decision_maker.topsis import TOPSIS

# item_criteria_matrix = [
#     [250, 16, 12, 5],
#     [200, 16, 8, 3],
#     [300, 32, 16, 4],
#     [275, 32, 8, 4],
#     [225, 16, 16, 2],
# ]

sla: SLA = SLA(discounted_cpu_threshold=0.5,
               discounted_memory_threshold=0.7,
               discounted_gpu_threshold=0.75)

cluster_state: ClusterState = ClusterState(node_number=2,
                                           sla=sla,
                                           available_cpu_units_per_node=500,
                                           available_memory_units_per_node=1000,
                                           available_gpu_units_per_node=100,
                                           cpu_unit_cost=10,
                                           memory_unit_cost=7,
                                           gpu_unit_cost=200)

topsis = TOPSIS(beneficial_column_indicator=[False, False, False, False],
                criteria_weights=[0.2, 0.15, 0.05, 0.6])

decision_maker = DecisionMaker(cluster_state=cluster_state, topsis=topsis)
decision = decision_maker.decide(total_future_required_cpu_units=275,
                                 total_future_required_memory_units=150,
                                 total_future_required_gpu_units=10)
print(decision)
