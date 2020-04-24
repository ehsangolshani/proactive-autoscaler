import torch

from denormalization import naive_denormalize
from workload_predictor.TCN.model import TCNModel, \
    load_and_initialize_tcn_workload_model
from workload_to_metric_mapper.DNN.model import DNNModel, \
    load_and_initialize_dnn_metrics_model
from torch.utils import data
from custom_datasets.windowed_dataset import WindowedDataset
from decision_maker.cluster_state import ClusterState
from decision_maker.decision import DecisionMaker
from decision_maker.sla import SLA
from decision_maker.topsis import TOPSIS

# load datasets
workload_prediction_model_window_size = 24

dataset = WindowedDataset(
    csv_path='raw_datasets/nasa-http/nasa_temporal_metrics_1m.csv',
    window_size=workload_prediction_model_window_size + 1
)

dataset_size = len(dataset)

data_loader: data.DataLoader = data.DataLoader(dataset=dataset,
                                               batch_size=1,
                                               num_workers=1,
                                               shuffle=True)

min_request_rate = dataset.min_request_rate
max_request_rate = dataset.max_request_rate
min_cpu_utilization = dataset.min_cpu_utilization
max_cpu_utilization = dataset.max_cpu_utilization
min_memory_utilization = dataset.min_memory_utilization
max_memory_utilization = dataset.max_memory_utilization
min_gpu_utilization = dataset.min_gpu_utilization
max_gpu_utilization = dataset.max_gpu_utilization

# load prediction models from file
workload_prediction_model: TCNModel = load_and_initialize_tcn_workload_model(
    model_path='trained_models/TCN_workload_model_nasa_dataset.pt',
    window_size=workload_prediction_model_window_size)
metrics_prediction_model: DNNModel = load_and_initialize_dnn_metrics_model(
    model_path='trained_models/DNN_metrics_model_nasa_dataset.pt')

# initial cluster state
sla: SLA = SLA(discounted_cpu_threshold=0.7,
               discounted_memory_threshold=0.7,
               discounted_gpu_threshold=0.7)

cluster_state: ClusterState = ClusterState(node_number=1,
                                           sla=sla,
                                           available_cpu_units_per_node=100,
                                           available_memory_units_per_node=60,
                                           available_gpu_units_per_node=25,
                                           cpu_unit_cost=4,
                                           memory_unit_cost=2,
                                           gpu_unit_cost=20)

topsis = TOPSIS(beneficial_column_indicator=[False, False, False, False, False, False, False],
                criteria_weights=[0.34, 0.24, 0.13, 0.14, 0.09, 0.03, 0.03])

decision_maker = DecisionMaker(cluster_state=cluster_state, topsis=topsis)

workload_prediction_model.eval()
metrics_prediction_model.eval()

print('\n\n')

for i, data in enumerate(data_loader, 0):

    if i > 15:
        break

    # predict workload
    previous_workload_sequence: torch.Tensor = data[:, 0:1, :-1]
    real_future_workload: torch.Tensor = data[:, 0:1, -1]
    real_future_workload = real_future_workload.view(-1)
    predicted_future_workload = workload_prediction_model(previous_workload_sequence)
    print('previous workload sequence: ', previous_workload_sequence)
    print('workload --> real: ', real_future_workload.item(),
          ' , predicted: ', predicted_future_workload.item())

    # predict metrics
    current_metrics: torch.Tensor = data[:, 0:4, -2]
    current_metrics[:, 0] = predicted_future_workload.item()
    real_future_metrics: torch.Tensor = data[:, 1:4, -1]
    predicted_future_metrics = metrics_prediction_model(current_metrics)
    print('metrics --> real: ', real_future_metrics,
          ' , predicted: ', predicted_future_metrics)

    # decision making
    real_future_metrics = real_future_metrics.view(-1)
    predicted_future_metrics = predicted_future_metrics.view(-1)

    denormalized_real_future_cpu = naive_denormalize(
        value=real_future_metrics[0].item(),
        minimum_value=min_cpu_utilization,
        maximum_value=max_cpu_utilization
    )

    denormalized_real_future_memory = naive_denormalize(
        value=real_future_metrics[1].item(),
        minimum_value=min_memory_utilization,
        maximum_value=max_memory_utilization
    )

    denormalized_real_future_gpu = naive_denormalize(
        value=real_future_metrics[2].item(),
        minimum_value=min_gpu_utilization,
        maximum_value=max_gpu_utilization
    )

    denormalized_predicted_future_cpu = naive_denormalize(
        value=predicted_future_metrics[0].item(),
        minimum_value=min_cpu_utilization,
        maximum_value=max_cpu_utilization
    )

    denormalized_predicted_future_memory = naive_denormalize(
        value=predicted_future_metrics[1].item(),
        minimum_value=min_memory_utilization,
        maximum_value=max_memory_utilization
    )

    denormalized_predicted_future_gpu = naive_denormalize(
        value=predicted_future_metrics[2].item(),
        minimum_value=min_gpu_utilization,
        maximum_value=max_gpu_utilization
    )

    decision = decision_maker.decide(
        total_future_required_cpu_units=denormalized_predicted_future_cpu,
        total_future_required_memory_units=denormalized_predicted_future_memory,
        total_future_required_gpu_units=denormalized_predicted_future_gpu
    )

    print('real resource --> cpu: {} , memory: {} , gpu: {}'
          .format(denormalized_real_future_cpu,
                  denormalized_real_future_memory,
                  denormalized_real_future_gpu))
    print('predicted resource --> cpu: {} , memory: {} , gpu: {}'
          .format(denormalized_predicted_future_cpu,
                  denormalized_predicted_future_memory,
                  denormalized_predicted_future_gpu))
    print('cluster state --> cpu: {} , memory: {} , gpu: {}'
          .format(cluster_state.calculate_total_available_cpu(),
                  cluster_state.calculate_total_available_memory(),
                  cluster_state.calculate_total_available_gpu()))

    action = decision[0]
    print('decision: ', decision)

    if action == 0:  # scale out
        cluster_state.scale_out()
    elif action == 2:  # scale down
        cluster_state.scale_down()

    print('cluster state after action --> cpu: {} , memory: {} , gpu: {}'
          .format(cluster_state.calculate_total_available_cpu(),
                  cluster_state.calculate_total_available_memory(),
                  cluster_state.calculate_total_available_gpu()))

    # calculate required/available proportion for metrics
    future_cpu_required_available_proportion = \
        denormalized_real_future_cpu / cluster_state.calculate_total_available_cpu()

    future_memory_required_available_proportion = \
        denormalized_real_future_memory / cluster_state.calculate_total_available_memory()

    future_gpu_required_available_proportion = \
        denormalized_real_future_gpu / cluster_state.calculate_total_available_gpu()

    # calculate violations
    future_cpu_violation = cluster_state.sla.calculate_cpu_violation(
        value=future_cpu_required_available_proportion)

    future_memory_violation = cluster_state.sla.calculate_memory_violation(
        value=future_memory_required_available_proportion)

    future_gpu_violation = cluster_state.sla.calculate_gpu_violation(
        value=future_gpu_required_available_proportion)

    # calculate under-utilization
    future_cpu_under_utilization = cluster_state.sla.calculate_cpu_under_utilization(
        value=future_cpu_required_available_proportion)

    future_memory_under_utilization = cluster_state.sla.calculate_memory_under_utilization(
        value=future_memory_required_available_proportion)

    future_gpu_under_utilization = cluster_state.sla.calculate_gpu_under_utilization(
        value=future_gpu_required_available_proportion)

    print('\n\n')
