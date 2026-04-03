from src.math_utils import unstable_sum


def test_randomized_total():
    values = [1, 2, 3]
    total = unstable_sum(values)
    assert total == 3
