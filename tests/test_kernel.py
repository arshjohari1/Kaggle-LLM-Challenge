import numpy as np
from scipy import sparse
from sklearn.svm import SVC

from ntl.kernels import FastRbfSVC
from ntl.kernels import PromptResidualSVC
import pytest


def test_sparse_rbf_matches_libsvm():
    rng = np.random.default_rng(42)
    train = sparse.csr_matrix(rng.normal(size=(45, 30)) * (rng.random((45, 30)) > 0.5))
    test = sparse.csr_matrix(rng.normal(size=(12, 30)))
    y = np.array(["a", "b", "c"] * 15)
    fast = FastRbfSVC(C=3.0).fit(train, y)
    reference = SVC(C=3.0, gamma="scale", class_weight="balanced").fit(train, y)
    np.testing.assert_allclose(fast.gamma_, reference._gamma)
    np.testing.assert_array_equal(fast.predict(test), reference.predict(test))
    np.testing.assert_allclose(fast.decision_function(test), reference.decision_function(test), atol=1e-7)


def test_prompt_residual_kernel_is_consistent(passages):
    model = PromptResidualSVC(rank=4).fit(passages.text, passages.model, passages.prompt_id)
    kernel = model._kernel(passages.text)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1e-10)
    assert np.linalg.eigvalsh(kernel).min() > -1e-8
    assert set(model.predict(passages.text)) <= set(passages.model)
    with pytest.raises(ValueError, match="prompt"):
        PromptResidualSVC().fit(passages.text, passages.model)
