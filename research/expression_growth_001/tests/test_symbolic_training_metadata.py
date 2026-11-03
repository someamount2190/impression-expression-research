import json

from experiments.impression_002e.streaming_training.train_and_verify import (
    validate_source_metadata,
)


def test_training_metadata_supports_variable_length_sft_and_custom_ids(tmp_path):
    token_ids = list(range(40, 72))
    sample = {
        "record_id": "eg_train",
        "token_ids": token_ids,
        "position_ids": list(range(len(token_ids))),
        "attention_mask": [1] * len(token_ids),
        "label_mask": [0] * 19 + [1] * 12,
    }
    held = {
        "record_id": "eg_val",
        "token_ids": token_ids[:20],
        "position_ids": list(range(20)),
        "attention_mask": [1] * 20,
        "label_mask": [0] * 9 + [1] * 10,
    }
    identity = {"cache_identity_sha256": "a" * 64}
    metadata = {
        "checkpoint_revision": "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364",
        "target_module": "model.layers.27.self_attn.q_proj",
        "rank": 4, "alpha": 8,
        "training_record_id": "eg_train", "heldout_record_id": "eg_val",
        "samples": {"eg_train": sample, "eg_val": held},
        "caches": {"eg_train": identity, "eg_val": {"cache_identity_sha256": "b" * 64}},
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    loaded, digest = validate_source_metadata(path)
    assert digest
    assert loaded["samples"]["eg_train"]["label_mask"] == sample["label_mask"]
