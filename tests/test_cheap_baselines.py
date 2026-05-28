from __future__ import annotations

import csv
import json
import pickle

import numpy as np

from crop.cheap_baselines import (
    COE_SCORE_COLUMNS,
    combine_npz_features,
    export_coe_answer_subset,
    export_coe_jsonl,
    export_text_npz,
    import_coe_answer_outputs,
    import_coe_outputs,
    load_crop_traces,
    merge_npz_files,
    oriented_error_scores,
    safe_inverse,
)
from crop.data import load_many_npz
from crop.experiments.common import build_score_bundle
from crop.splits import split_traces


def _tiny_annotated(path):
    rows = [
        {
            "expression_id": "0",
            "original_expression": "1 + 2",
            "correct_value": "3",
            "predicted_value": "3",
            "total_steps": 2,
            "step_expressions": [
                {"step_number": 0, "step_content": "Add 1 and 2.", "step_label": True},
                {"step_number": 1, "step_content": "The result is 4.", "step_label": False},
            ],
        },
        {
            "expression_id": "1",
            "original_expression": "2 + 2",
            "correct_value": "4",
            "predicted_value": "4",
            "total_steps": 1,
            "step_expressions": [
                {"step_number": 0, "step_content": "The result is 4.", "step_label": True},
            ],
        },
    ]
    path.write_text(json.dumps(rows))


def test_export_coe_jsonl_from_crop_annotations(tmp_path):
    annotated = tmp_path / "arith.nt3.annotated.json"
    _tiny_annotated(annotated)
    traces = load_crop_traces([annotated])
    paths = export_coe_jsonl(traces, tmp_path, dataset_prefix="tiny")

    step_lines = [json.loads(line) for line in paths["step_jsonl"].read_text().splitlines()]
    trace_lines = [json.loads(line) for line in paths["trace_jsonl"].read_text().splitlines()]
    assert len(step_lines) == 3
    assert len(trace_lines) == 2
    assert step_lines[1]["cached_output"] == "The result is 4."
    assert "Reasoning so far" in step_lines[1]["en"]

    with paths["step_manifest"].open(newline="") as f:
        manifest = list(csv.DictReader(f))
    assert manifest[1]["y_error"] == "1"
    assert manifest[1]["trace_id"].startswith("arithmetic:")


def test_import_coe_outputs_and_score_orientation(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "dataset,dataset_index,granularity,trace_id,domain,complexity,source_file,source_stem,"
        "expression_id,step_number,total_steps,step_label,y_error,original_expression,correct_value,"
        "predicted_value,step_content\n"
        "tiny,0,step,trace0,arithmetic,3,x,src,0,0,1,False,1,expr,3,4,bad step\n"
    )
    root = tmp_path / "coe"
    out_dir = root / "OutputInfo/en/Output/Llama-3.1-8B-Instruct/tiny"
    coe_dir = root / "OutputInfo/en/CoE/Llama-3.1-8B-Instruct/tiny"
    cotk_dir = root / "OutputInfo/en/CoTK/Llama-3.1-8B-Instruct/tiny"
    out_dir.mkdir(parents=True)
    coe_dir.mkdir(parents=True)
    cotk_dir.mkdir(parents=True)
    with (out_dir / "tiny_0.pkl").open("wb") as f:
        pickle.dump({"maxprob": 0.5, "ppl": 2.0, "entropy": 3.0, "tempscl": 0.25, "energy": -4.0}, f)
    with (coe_dir / "tiny.average_0.pkl").open("wb") as f:
        pickle.dump({"R": 0.1, "C": 0.2}, f)
    with (cotk_dir / "tiny.average_0.pkl").open("wb") as f:
        pickle.dump({"CoTK": 0.4}, f)

    output = import_coe_outputs(root, manifest, tmp_path / "scores.npz")
    with np.load(output, allow_pickle=True) as data:
        assert data["features"].shape == (1, len(COE_SCORE_COLUMNS))
        names = list(data["feature_names"])
        values = dict(zip(names, data["features"][0]))
    assert values["maxprob_error"] == 2.0
    assert values["ppl_error"] == 2.0
    assert values["tempscl_error"] == 4.0
    assert values["cotk_error"] == 2.5
    assert safe_inverse(0.0) == 1e8

    scores = oriented_error_scores(
        {"maxprob": 0.5, "ppl": 2.0, "entropy": 3.0, "tempscl": 0.25, "energy": -4.0},
        {"R": 0.1, "C": 0.2},
        {"CoTK": 0.4},
    )
    assert set(scores) == set(COE_SCORE_COLUMNS)


def test_column_score_source_loads_named_feature(tmp_path):
    metadata = []
    features = []
    for i, y_error in enumerate([0, 1, 0, 1, 0, 1, 0, 1]):
        metadata.append(
            {
                "expr_id": f"t{i}",
                "trace_id": f"t{i}",
                "domain": "toy",
                "step_number": 0,
                "total_steps": 1,
                "before_after": "step",
                "step_labels": {"step_number": 0, "step_content": "x", "step_label": not y_error},
                "original_expression": "x",
            }
        )
        features.append([0.1 if y_error == 0 else 0.9])
    path = tmp_path / "scores.npz"
    np.savez(
        path,
        features=np.asarray(features, dtype=float),
        metadata=np.asarray(metadata, dtype=object),
        feature_names=np.asarray(["score_error"], dtype=object),
    )
    traces = load_many_npz([path], ["mixed"])
    split = split_traces(traces, seed=0)
    bundle = build_score_bundle("column:score_error", split, seed=0)
    assert bundle.cal_step_scores.min() >= 0.0
    assert bundle.test_step_scores.max() <= 1.0


def test_trace_text_features_do_not_include_label(tmp_path):
    annotated = tmp_path / "arith.nt3.annotated.json"
    _tiny_annotated(annotated)
    traces = load_crop_traces([annotated])
    output = export_text_npz(traces, tmp_path / "trace_text.npz", granularity="trace")
    with np.load(output, allow_pickle=True) as data:
        features = data["features"]
    assert features.shape == (2, 55)
    assert not np.array_equal(features[:, 4], np.asarray([int(t.has_error) for t in traces]))


def test_merge_npz_files_preserves_feature_names_and_rows(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez(
        first,
        features=np.asarray([[1.0, 2.0]]),
        metadata=np.asarray([{"trace_id": "a"}], dtype=object),
        feature_names=np.asarray(["x", "y"], dtype=object),
    )
    np.savez(
        second,
        features=np.asarray([[3.0, 4.0], [5.0, 6.0]]),
        metadata=np.asarray([{"trace_id": "b"}, {"trace_id": "c"}], dtype=object),
        feature_names=np.asarray(["x", "y"], dtype=object),
    )

    output = merge_npz_files([first, second], tmp_path / "merged.npz")
    with np.load(output, allow_pickle=True) as data:
        assert data["features"].shape == (3, 2)
        assert list(data["feature_names"]) == ["x", "y"]
        assert [row["trace_id"] for row in data["metadata"]] == ["a", "b", "c"]


def test_combine_npz_features_aligns_by_trace_and_step(tmp_path):
    left_meta = [
        {"trace_id": "a", "step_number": 0, "step_labels": {"step_label": True}},
        {"trace_id": "a", "step_number": 1, "step_labels": {"step_label": False}},
    ]
    right_meta = [
        {"trace_id": "a", "step_number": 1, "step_labels": {"step_label": False}},
        {"trace_id": "a", "step_number": 0, "step_labels": {"step_label": True}},
    ]
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    np.savez(
        left,
        features=np.asarray([[1.0], [2.0]]),
        metadata=np.asarray(left_meta, dtype=object),
        feature_names=np.asarray(["text"], dtype=object),
    )
    np.savez(
        right,
        features=np.asarray([[20.0], [10.0]]),
        metadata=np.asarray(right_meta, dtype=object),
        feature_names=np.asarray(["score"], dtype=object),
    )
    output = combine_npz_features(left, right, tmp_path / "combined.npz", granularity="step")
    with np.load(output, allow_pickle=True) as data:
        assert list(data["feature_names"]) == ["text", "score"]
        assert data["features"].tolist() == [[1.0, 10.0], [2.0, 20.0]]


def test_export_and_import_coe_answer_outputs(tmp_path, monkeypatch):
    coe_data = tmp_path / "Data"
    coe_data.mkdir()
    (coe_data / "commonsenseqa.jsonl").write_text(
        json.dumps({"id": 7, "en": "Question: x\nChoices:\n(A) yes\n(B) no", "answer": "A"}) + "\n"
    )
    paths = export_coe_answer_subset(coe_data, tmp_path / "answer_jsonl", datasets=["commonsenseqa"])
    manifest = paths["commonsenseqa_manifest"]
    with manifest.open(newline="") as f:
        row = next(csv.DictReader(f))

    root = tmp_path / "coe"
    eval_dir = root / "Evaluation"
    eval_dir.mkdir(parents=True)
    (eval_dir / "__init__.py").write_text("")
    (eval_dir / "match.py").write_text(
        "class AnswerParsing:\n"
        "    def __init__(self, dataset):\n"
        "        self.dataset = dataset\n"
        "    def dataset_parse(self, pred, true, sample):\n"
        "        return 'A', 'Answer: A' in pred and true == 'A'\n"
    )
    dataset = row["dataset"]
    out_dir = root / "OutputInfo/en/Output/Llama-3.1-8B-Instruct" / dataset
    coe_dir = root / "OutputInfo/en/CoE/Llama-3.1-8B-Instruct" / dataset
    cotk_dir = root / "OutputInfo/en/CoTK/Llama-3.1-8B-Instruct" / dataset
    out_dir.mkdir(parents=True)
    coe_dir.mkdir(parents=True)
    cotk_dir.mkdir(parents=True)
    output_text = "Reasoning. Answer: A<|eot_id|>"
    with (out_dir / f"{dataset}_0.pkl").open("wb") as f:
        pickle.dump(
            {
                "output_seq": output_text,
                "input_seq": "prompt",
                "maxprob": 0.5,
                "ppl": 2.0,
                "entropy": 3.0,
                "tempscl": 0.25,
                "energy": -4.0,
            },
            f,
        )
    with (coe_dir / f"{dataset}.average_0.pkl").open("wb") as f:
        pickle.dump({"R": 0.1, "C": 0.2}, f)
    with (cotk_dir / f"{dataset}.average_0.pkl").open("wb") as f:
        pickle.dump({"CoTK": 0.4}, f)
    output = import_coe_answer_outputs(
        root,
        [manifest],
        tmp_path / "answer_scores.npz",
        text_npz=tmp_path / "answer_text.npz",
    )
    with np.load(output, allow_pickle=True) as data:
        assert data["features"].shape == (1, len(COE_SCORE_COLUMNS))
        meta = data["metadata"][0]
        assert meta["y_error"] == 0
        assert meta["step_labels"]["step_label"] is True
    with np.load(tmp_path / "answer_text.npz", allow_pickle=True) as data:
        assert data["features"].shape == (1, 55)
