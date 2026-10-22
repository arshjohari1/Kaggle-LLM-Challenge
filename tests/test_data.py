import numpy as np
import pandas as pd
import pytest

from ntl.data import groups, make_folds, read_data, validate_folds


def test_folds_reproducible_and_prompt_disjoint(passages):
    a = make_folds(passages)
    pd.testing.assert_frame_equal(a, make_folds(passages))
    folds = validate_folds(passages, a)
    assert len(set(folds)) == 5
    for f in range(5):
        assert not set(passages.prompt_id[folds == f]) & set(passages.prompt_id[folds != f])


def test_duplicate_connectivity_and_leakage_rejected(passages):
    passages.loc[18, "text"] = passages.loc[0, "text"]
    passages.loc[36, "text"] = passages.loc[0, "text"] + "  "
    connected = groups(passages)
    assert connected[0] == connected[18] == connected[36]
    folds = make_folds(passages, 3)
    validate_folds(passages, folds)
    folds.loc[18, "fold"] = (folds.loc[0, "fold"] + 1) % 3
    with pytest.raises(ValueError, match="leakage"):
        validate_folds(passages, folds)


def test_schema_and_preserved_text(passages, tmp_path):
    p = tmp_path / "train.csv"
    passages.loc[0, "id"] = "00001"
    passages.loc[0, "text"] = "  NA\nHeading: Mixed CASE!  "
    passages.to_csv(p, index=False)
    result = read_data(p)
    assert result.loc[0, "id"] == "00001"
    assert result.loc[0, "text"] == passages.loc[0, "text"]
    passages.loc[1, "id"] = "00001"
    passages.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Duplicate IDs"):
        read_data(p)


def test_invalid_labels_and_blank_text(passages, tmp_path):
    p = tmp_path / "bad.csv"
    passages.loc[0, "text"] = " "
    passages.to_csv(p, index=False)
    with pytest.raises(ValueError, match="text"):
        read_data(p)
    passages.loc[0, "text"] = "Okay"
    passages.loc[0, "model"] = "unknown-model"
    passages.to_csv(p, index=False)
    with pytest.raises(ValueError, match="unknown"):
        read_data(p)


def test_fold_ids_must_match(passages):
    folds = make_folds(passages)
    folds.loc[0, "id"] = "wrong"
    with pytest.raises(ValueError, match="IDs"):
        validate_folds(passages, folds)
