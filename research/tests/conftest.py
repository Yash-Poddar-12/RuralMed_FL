import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(1)
