from decision_maker.sla import SLA


class ClusterState:
    def __init__(self, node_number: int, sla: SLA,
                 available_cpu_units_per_node: float,
                 available_memory_units_per_node: float,
                 available_gpu_units_per_node: float,
                 cpu_unit_cost: float, memory_unit_cost: float, gpu_unit_cost: float):
        self.node_number: int = node_number
        self.sla: SLA = sla
        self.available_cpu_units_per_node: float = available_cpu_units_per_node
        self.available_memory_units_per_node: float = available_memory_units_per_node
        self.available_gpu_units_per_node: float = available_gpu_units_per_node
        self.cpu_unit_cost: float = cpu_unit_cost
        self.memory_unit_cost: float = memory_unit_cost
        self.gpu_unit_cost: float = gpu_unit_cost

    def calculate_total_available_cpu(self, scale_out=False, scale_down=False):
        if not scale_out and not scale_down:
            return self.node_number * self.available_cpu_units_per_node
        elif scale_out:
            return (self.node_number + 1) * self.available_cpu_units_per_node
        elif scale_down:
            return (self.node_number - 1) * self.available_cpu_units_per_node

    def calculate_total_available_memory(self, scale_out=False, scale_down=False):
        if not scale_out and not scale_down:
            return self.node_number * self.available_memory_units_per_node
        elif scale_out:
            return (self.node_number + 1) * self.available_memory_units_per_node
        elif scale_down:
            return (self.node_number - 1) * self.available_memory_units_per_node

    def calculate_total_available_gpu(self, scale_out=False, scale_down=False):
        if not scale_out and not scale_down:
            return self.node_number * self.available_gpu_units_per_node
        elif scale_out:
            return (self.node_number + 1) * self.available_gpu_units_per_node
        elif scale_down:
            return (self.node_number - 1) * self.available_gpu_units_per_node

    def calculate_cpu_cost(self, scale_out=False, scale_down=False):
        if not scale_out and not scale_down:
            return self.node_number * self.available_cpu_units_per_node * self.cpu_unit_cost
        elif scale_out:
            return (self.node_number + 1) * self.available_cpu_units_per_node * self.cpu_unit_cost
        elif scale_down:
            return (self.node_number - 1) * self.available_cpu_units_per_node * self.cpu_unit_cost

    def calculate_memory_cost(self, scale_out=False, scale_down=False):
        if not scale_out and not scale_down:
            return self.node_number * self.available_memory_units_per_node * self.memory_unit_cost
        elif scale_out:
            return (self.node_number + 1) * self.available_memory_units_per_node * self.memory_unit_cost
        elif scale_down:
            return (self.node_number - 1) * self.available_memory_units_per_node * self.memory_unit_cost

    def calculate_gpu_cost(self, scale_out=False, scale_down=False):
        if not scale_out and not scale_down:
            return self.node_number * self.available_gpu_units_per_node * self.gpu_unit_cost
        elif scale_out:
            return (self.node_number + 1) * self.available_gpu_units_per_node * self.gpu_unit_cost
        elif scale_down:
            return (self.node_number - 1) * self.available_gpu_units_per_node * self.gpu_unit_cost

    def calculate_total_resource_cost(self, scale_out=False, scale_down=False):
        return self.calculate_cpu_cost(scale_out, scale_down) + \
               self.calculate_memory_cost(scale_out, scale_down) + \
               self.calculate_gpu_cost(scale_out, scale_down)

    def scale_out(self):
        self.node_number += 1

    def scale_down(self):
        self.node_number -= 1
