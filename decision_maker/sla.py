class SLA:
    # argument should be between 0 and 1
    def __init__(self, discounted_cpu_threshold: float,
                 discounted_memory_threshold: float,
                 discounted_gpu_threshold: float):
        self.discounted_cpu_threshold = discounted_cpu_threshold
        self.discounted_memory_threshold = discounted_memory_threshold
        self.discounted_gpu_threshold = discounted_gpu_threshold

    def calculate_cpu_violation(self, value: float):
        violation = value - self.discounted_cpu_threshold
        return violation if violation > 0 else 0

    def calculate_memory_violation(self, value: float):
        violation = value - self.discounted_memory_threshold
        return violation if violation > 0 else 0

    def calculate_gpu_violation(self, value: float):
        violation = value - self.discounted_gpu_threshold
        return violation if violation > 0 else 0

    def calculate_cpu_under_utilization(self, value: float):
        under_utilization = self.discounted_cpu_threshold - value
        return under_utilization if under_utilization > 0 else 0

    def calculate_memory_under_utilization(self, value: float):
        under_utilization = self.discounted_memory_threshold - value
        return under_utilization if under_utilization > 0 else 0

    def calculate_gpu_under_utilization(self, value: float):
        under_utilization = self.discounted_gpu_threshold - value
        return under_utilization if under_utilization > 0 else 0
