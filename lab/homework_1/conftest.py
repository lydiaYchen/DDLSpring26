import pytest
from testbook import testbook


def pytest_addoption(parser):
    parser.addoption("--fpath", action="store")


@pytest.fixture(scope="session")
def tb(request):
    fpath = request.config.getoption("--fpath")

    with testbook(fpath, execute=False) as tb_:
        yield tb_
