"""Regression coverage for the platform rounding observed on GitHub Actions."""
import pytest
from verify_snapshot import first_differences


def test_ci_angle_roundoff_is_accepted():
    assert not first_differences(
        {'angle': 75.00000004218944, 'vertices': [1, 2]},
        {'angle': 75.00000004218946, 'vertices': [1, 2]})


@pytest.mark.parametrize('actual,expected', [
    ({'angle': 75.00001}, {'angle': 75.0}),
    ({'vertices': [1, 3]}, {'vertices': [1, 2]}),
    ({'vertices': [1.0, 2]}, {'vertices': [1, 2]}),
    ({'vertices': [1]}, {'vertices': [1, 2]}),
    ({'id': 1}, {'other_id': 1}),
])
def test_material_or_discrete_changes_are_rejected(actual, expected):
    assert first_differences(actual, expected)
