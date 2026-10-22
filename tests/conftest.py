import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from ntl.data import LABELS


@pytest.fixture(autouse=True)
def bounded_test_threads():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def passages():
    rows = []
    for prompt in range(10):
        for label_i, label in enumerate(LABELS):
            text = (f"Topic {prompt}: This is a synthetic passage. "
                    f"Author marker{label_i} uses characteristic token{label_i}! " * 4)
            rows.append({"id": f"{prompt:02}-{label_i:02}", "prompt_id": str(prompt),
                         "text": text, "model": label})
    return pd.DataFrame(rows)
