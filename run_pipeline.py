from datetime import datetime

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
import matplotlib.pyplot as plt

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
                                               shuffle=False)

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

cluster_state: ClusterState = ClusterState(node_number=2,
                                           sla=sla,
                                           available_cpu_units_per_node=100,
                                           available_memory_units_per_node=60,
                                           available_gpu_units_per_node=25,
                                           cpu_unit_cost=4,
                                           memory_unit_cost=2,
                                           gpu_unit_cost=20)

# thesis configuration
# topsis = TOPSIS(beneficial_column_indicator=[False, False, False, False, False, False, False],
#                 criteria_weights=[0.34, 0.24, 0.13, 0.14, 0.09, 0.03, 0.03])

topsis = TOPSIS(beneficial_column_indicator=[False, False, False, False, False, False, False],
                criteria_weights=[0.14, 0.09, 0.03, 0.34, 0.24, 0.13, 0.03])

decision_maker = DecisionMaker(cluster_state=cluster_state, topsis=topsis)

workload_prediction_model.eval()
metrics_prediction_model.eval()

# variables to draw some plots
avg_metrics_x = list()
avg_cpu_violation_y = list()
avg_memory_violation_y = list()
avg_gpu_violation_y = list()
avg_cpu_under_utilization_y = list()
avg_memory_under_utilization_y = list()
avg_gpu_under_utilization_y = list()
avg_cost_y = list()
avg_node_num_y = list()

sum_of_cpu_violation_for_plot = 0
sum_of_memory_violation_for_plot = 0
sum_of_gpu_violation_for_plot = 0

sum_of_cpu_under_utilization_for_plot = 0
sum_of_memory_under_utilization_for_plot = 0
sum_of_gpu_under_utilization_for_plot = 0

sum_of_cost_for_plot = 0
sum_of_node_num_for_plot = 0

plot_x_counter = 0

# variables to calculate some statistics over all the data
sum_of_cpu_violation = 0
sum_of_memory_violation = 0
sum_of_gpu_violation = 0

sum_of_cpu_under_utilization = 0
sum_of_memory_under_utilization = 0
sum_of_gpu_under_utilization = 0

sum_of_cost = 0
sum_of_node_num = 0
##########

iteration = 0

comp1_response_time_sum = 0
comp1_response_time_counter = 0

comp2_response_time_sum = 0
comp2_response_time_counter = 0

comp3_response_time_sum = 0
comp3_response_time_counter = 0

for i, data in enumerate(data_loader, 0):

    iteration += 1

    # predict workload
    comp1_start_timestamp = datetime.now().timestamp()

    previous_workload_sequence: torch.Tensor = data[:, 0:1, :-1]
    real_future_workload: torch.Tensor = data[:, 0:1, -1]
    real_future_workload = real_future_workload.view(-1)
    predicted_future_workload = workload_prediction_model(previous_workload_sequence)

    comp1_finish_timestamp = datetime.now().timestamp()
    comp1_diff_in_seconds = comp1_finish_timestamp - comp1_start_timestamp
    comp1_response_time_counter += 1
    comp1_response_time_sum += comp1_diff_in_seconds

    print('workload --> real: ', real_future_workload.item(),
          ' , predicted: ', predicted_future_workload.item())

    # predict metrics
    comp2_start_timestamp = datetime.now().timestamp()

    current_metrics: torch.Tensor = data[:, 0:4, -2]
    current_metrics[:, 0] = predicted_future_workload.item()
    real_future_metrics: torch.Tensor = data[:, 1:4, -1]
    predicted_future_metrics = metrics_prediction_model(current_metrics)

    comp2_finish_timestamp = datetime.now().timestamp()
    comp2_diff_in_seconds = comp2_finish_timestamp - comp2_start_timestamp
    comp2_response_time_counter += 1
    comp2_response_time_sum += comp2_diff_in_seconds

    print('metrics --> real: ', real_future_metrics,
          ' , predicted: ', predicted_future_metrics)

    # decision making
    comp3_start_timestamp = datetime.now().timestamp()

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

    comp3_finish_timestamp = datetime.now().timestamp()
    comp3_diff_in_seconds = comp3_finish_timestamp - comp3_start_timestamp
    comp3_response_time_counter += 1
    comp3_response_time_sum += comp3_diff_in_seconds

    print('cluster state --> cpu: {} , memory: {} , gpu: {}'
          .format(cluster_state.calculate_total_available_cpu(),
                  cluster_state.calculate_total_available_memory(),
                  cluster_state.calculate_total_available_gpu()))
    print('predicted resource --> cpu: {} , memory: {} , gpu: {}'
          .format(denormalized_predicted_future_cpu,
                  denormalized_predicted_future_memory,
                  denormalized_predicted_future_gpu))

    action = decision[0]
    print('decision: ', decision)

    if action == 0:  # scale out
        cluster_state.scale_out()
    elif action == 2:  # scale down
        cluster_state.scale_down()

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

    # calculate cost
    future_cost = cluster_state.calculate_total_resource_cost()

    # sum up
    sum_of_cpu_violation += future_cpu_violation
    sum_of_memory_violation += future_memory_violation
    sum_of_gpu_violation += future_gpu_violation
    sum_of_cpu_under_utilization += future_cpu_under_utilization
    sum_of_memory_under_utilization += future_memory_under_utilization
    sum_of_gpu_under_utilization += future_gpu_under_utilization
    sum_of_cost += future_cost
    sum_of_node_num += cluster_state.node_number

    sum_of_cpu_violation_for_plot += future_cpu_violation
    sum_of_memory_violation_for_plot += future_memory_violation
    sum_of_gpu_violation_for_plot += future_gpu_violation
    sum_of_cpu_under_utilization_for_plot += future_cpu_under_utilization
    sum_of_memory_under_utilization_for_plot += future_memory_under_utilization
    sum_of_gpu_under_utilization_for_plot += future_gpu_under_utilization
    sum_of_cost_for_plot += future_cost
    sum_of_node_num_for_plot += cluster_state.node_number

    if iteration % 100 == 0:
        plot_x_counter += 1
        avg_metrics_x.append(plot_x_counter)

        denominator = 100

        avg_cpu_violation_y.append(sum_of_cpu_violation_for_plot / denominator)
        avg_memory_violation_y.append(sum_of_memory_violation_for_plot / denominator)
        avg_gpu_violation_y.append(sum_of_gpu_violation_for_plot / denominator)
        avg_cpu_under_utilization_y.append(sum_of_cpu_under_utilization_for_plot / denominator)
        avg_memory_under_utilization_y.append(sum_of_memory_under_utilization_for_plot / denominator)
        avg_gpu_under_utilization_y.append(sum_of_gpu_under_utilization_for_plot / denominator)
        avg_cost_y.append(sum_of_cost_for_plot / denominator)
        avg_node_num_y.append(sum_of_node_num_for_plot / denominator)

        sum_of_cpu_violation_for_plot = 0
        sum_of_memory_violation_for_plot = 0
        sum_of_gpu_violation_for_plot = 0
        sum_of_cpu_under_utilization_for_plot = 0
        sum_of_memory_under_utilization_for_plot = 0
        sum_of_gpu_under_utilization_for_plot = 0
        sum_of_cost_for_plot = 0
        sum_of_node_num_for_plot = 0

    print('\n\n')

denominator = 100

plot_x_where_train_stopped = iteration / 100
iteration_where_train_stopped = iteration

avg_metrics_x.append(plot_x_where_train_stopped)

avg_cpu_violation_y.append(sum_of_cpu_violation_for_plot / (iteration % denominator))
avg_memory_violation_y.append(sum_of_memory_violation_for_plot / (iteration % denominator))
avg_gpu_violation_y.append(sum_of_gpu_violation_for_plot / (iteration % denominator))
avg_cpu_under_utilization_y.append(sum_of_cpu_under_utilization_for_plot / (iteration % denominator))
avg_memory_under_utilization_y.append(sum_of_memory_under_utilization_for_plot / (iteration % denominator))
avg_gpu_under_utilization_y.append(sum_of_gpu_under_utilization_for_plot / (iteration % denominator))
avg_cost_y.append(sum_of_cost_for_plot / (iteration % denominator))
avg_node_num_y.append(sum_of_node_num_for_plot / (iteration % denominator))

sum_of_cpu_violation_for_plot = 0
sum_of_memory_violation_for_plot = 0
sum_of_gpu_violation_for_plot = 0
sum_of_cpu_under_utilization_for_plot = 0
sum_of_memory_under_utilization_for_plot = 0
sum_of_gpu_under_utilization_for_plot = 0
sum_of_cost_for_plot = 0
sum_of_node_num_for_plot = 0

print('\n\n')

print("average CPU violation: ", sum_of_cpu_violation / dataset_size)
print("average Memory violation: ", sum_of_memory_violation / dataset_size)
print("average GPU violation: ", sum_of_gpu_violation / dataset_size)

print("average CPU under-utilization: ", sum_of_cpu_under_utilization / dataset_size)
print("average Memory under-utilization: ", sum_of_memory_under_utilization / dataset_size)
print("average GPU under-utilization: ", sum_of_gpu_under_utilization / dataset_size)

print("average Cost: ", sum_of_cost / dataset_size)
print("average Number of Nodes: ", sum_of_node_num / dataset_size)

print('average response time of workload predictor: ', (comp1_response_time_sum * 1000) / comp1_response_time_counter)
print('average response time of resource predictor: ', (comp2_response_time_sum * 1000) / comp2_response_time_counter)
print('average response time of decision maker: ', (comp3_response_time_sum * 1000) / comp3_response_time_counter)

# draw plots

plt.figure(figsize=[12.0, 8.0])
plt.title('SLA CPU Violation')
plt.xlabel("Time")
plt.ylabel("CPU Violation")
plt.axis([0, dataset_size / 100 + 1, 0, 1.5])
plt.plot(avg_metrics_x, avg_cpu_violation_y, 'r-')
plt.savefig('cpu_violation_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('SLA Memory Violation')
plt.xlabel("Time")
plt.ylabel("Memory Violation")
plt.axis([0, dataset_size / 100 + 1, 0, 1.5])
plt.plot(avg_metrics_x, avg_memory_violation_y, 'r-')
plt.savefig('memory_violation_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('SLA GPU Violation')
plt.xlabel("Time")
plt.ylabel("GPU Violation")
plt.axis([0, dataset_size / 100 + 1, 0, 1.5])
plt.plot(avg_metrics_x, avg_gpu_violation_y, 'r-')
plt.savefig('gpu_violation_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('CPU under-utilization')
plt.xlabel("Time")
plt.ylabel("CPU under-utilization")
plt.axis([0, dataset_size / 100 + 1, 0, 1.5])
plt.plot(avg_metrics_x, avg_cpu_under_utilization_y, 'b-')
plt.savefig('cpu_under_utilization_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('Memory under-utilization')
plt.xlabel("Time")
plt.ylabel("Memory under-utilization")
plt.axis([0, dataset_size / 100 + 1, 0, 1.5])
plt.plot(avg_metrics_x, avg_memory_under_utilization_y, 'b-')
plt.savefig('memory_under_utilization_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('GPU under-utilization')
plt.xlabel("Time")
plt.ylabel("GPU under-utilization")
plt.axis([0, dataset_size / 100 + 1, 0, 1.5])
plt.plot(avg_metrics_x, avg_gpu_under_utilization_y, 'b-')
plt.savefig('gpu_under_utilization_plot.png')
plt.show()
plt.close()

maximum_resource_cost = (max_cpu_utilization + 10) * cluster_state.cpu_unit_cost \
                        + (max_memory_utilization + 10) * cluster_state.memory_unit_cost \
                        + (max_gpu_utilization + 10) * cluster_state.gpu_unit_cost

plt.figure(figsize=[12.0, 8.0])
plt.title('Average Resource Cost')
plt.xlabel("Time")
plt.ylabel("Cost")
plt.axis([0, dataset_size / 100 + 1, 0, maximum_resource_cost])
plt.plot(avg_metrics_x, avg_cost_y, 'g-')
plt.savefig('resource_cost_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('Average Number of Nodes')
plt.xlabel("Time")
plt.ylabel("Number of Nodes")
plt.axis([0, dataset_size / 100 + 1, 0, 30])
plt.plot(avg_metrics_x, avg_node_num_y, 'g-')
plt.savefig('node_num_plot.png')
plt.show()
plt.close()
