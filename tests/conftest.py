import pytest
import torch


@pytest.fixture
def smooth_image():
    """A smooth algebraic test input independent of downloaded MRI data."""

    def make(shape):
        y = torch.linspace(0, torch.pi, shape[0]).sin().square()
        x = torch.linspace(0, torch.pi, shape[1]).sin().square()
        return y[:, None] * x[None, :]

    return make


@pytest.fixture(scope="session", autouse=True)
def small_cpu_thread_pool():
    """Avoid oversubscribing many cores for the small reference matrices."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
