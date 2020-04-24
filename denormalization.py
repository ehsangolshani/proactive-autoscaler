def naive_denormalize(value, minimum_value, maximum_value):
    return minimum_value + value * (maximum_value - minimum_value) \
        if value > 0 else minimum_value
