#!/usr/bin/env python3
"""Convert a BERT-style RuBERT checkpoint into a Longformer and continue MLM pretraining."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _get_base_model(source_model):
    base_model_prefix = getattr(source_model, "base_model_prefix", "")
    base_model = getattr(source_model, base_model_prefix, None)
    if base_model is None:
        raise TypeError(
            "Only BERT-style masked language models are supported. "
            f"Could not resolve base model via base_model_prefix={base_model_prefix!r}."
        )
    return base_model


def _load_tokenizer(model_ref: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )


def _build_longformer_config(source_config, tokenizer, target_max_length: int, attention_window: int):
    from transformers import LongformerConfig

    if attention_window <= 0 or attention_window % 2 != 0:
        raise ValueError("--attention-window must be a positive even integer.")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = getattr(source_config, "pad_token_id", 0)

    cls_token_id = tokenizer.cls_token_id
    if cls_token_id is None:
        cls_token_id = getattr(source_config, "bos_token_id", 0)

    sep_token_id = tokenizer.sep_token_id
    if sep_token_id is None:
        sep_token_id = getattr(source_config, "sep_token_id", cls_token_id)

    max_position_embeddings = target_max_length + pad_token_id + 1
    return LongformerConfig(
        attention_window=[attention_window] * int(source_config.num_hidden_layers),
        vocab_size=int(source_config.vocab_size),
        hidden_size=int(source_config.hidden_size),
        num_hidden_layers=int(source_config.num_hidden_layers),
        num_attention_heads=int(source_config.num_attention_heads),
        intermediate_size=int(source_config.intermediate_size),
        hidden_act=source_config.hidden_act,
        hidden_dropout_prob=float(source_config.hidden_dropout_prob),
        attention_probs_dropout_prob=float(source_config.attention_probs_dropout_prob),
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=int(getattr(source_config, "type_vocab_size", 2)),
        initializer_range=float(source_config.initializer_range),
        layer_norm_eps=float(source_config.layer_norm_eps),
        pad_token_id=int(pad_token_id),
        bos_token_id=int(cls_token_id),
        eos_token_id=int(sep_token_id),
        sep_token_id=int(sep_token_id),
    )


def _expand_position_embeddings(source_weight, target_weight, padding_idx: int):
    expanded = target_weight.detach().clone()
    source_max_positions = source_weight.size(0)
    copy_start = int(padding_idx) + 1
    copy_cursor = copy_start

    while copy_cursor < expanded.size(0):
        chunk = min(source_max_positions, expanded.size(0) - copy_cursor)
        expanded[copy_cursor:copy_cursor + chunk] = source_weight[:chunk]
        copy_cursor += chunk

    return expanded


def _copy_bert_mlm_head(source_model, target_model) -> None:
    source_head = getattr(source_model, "cls", None)
    if source_head is None or not hasattr(source_head, "predictions"):
        raise TypeError(
            "Expected a BERT-style MLM head (`model.cls.predictions`) on the source checkpoint."
        )

    source_predictions = source_head.predictions
    target_head = target_model.lm_head

    target_head.dense.load_state_dict(source_predictions.transform.dense.state_dict())
    target_head.layer_norm.load_state_dict(source_predictions.transform.LayerNorm.state_dict())
    target_head.decoder.weight.data.copy_(source_predictions.decoder.weight.data)

    if hasattr(source_predictions, "bias") and hasattr(target_head, "bias"):
        target_head.bias.data.copy_(source_predictions.bias.data)
        target_head.decoder.bias = target_head.bias
    elif getattr(source_predictions.decoder, "bias", None) is not None:
        target_head.bias.data.copy_(source_predictions.decoder.bias.data)
        target_head.decoder.bias = target_head.bias


def convert_rubert_to_longformer(
    source_model_name: str,
    output_dir: str | Path,
    target_max_length: int = 4096,
    attention_window: int = 512,
    trust_remote_code: bool = False,
    seed: int = 42,
):
    set_global_seed(seed)

    try:
        import torch
        from transformers import (
            AutoModelForMaskedLM,
            LongformerForMaskedLM,
        )
    except ImportError as exc:
        raise ImportError(
            "Install `transformers`, `torch` and `accelerate` to use rubert_longformer.py."
        ) from exc

    if target_max_length <= 0:
        raise ValueError("--max-length must be positive.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(
        source_model_name,
        trust_remote_code=trust_remote_code,
    )
    source_model = AutoModelForMaskedLM.from_pretrained(
        source_model_name,
        trust_remote_code=trust_remote_code,
    )
    base_model = _get_base_model(source_model)

    longformer_config = _build_longformer_config(
        source_model.config,
        tokenizer,
        target_max_length=target_max_length,
        attention_window=attention_window,
    )
    longformer_model = LongformerForMaskedLM(longformer_config)
    longformer_base = longformer_model.longformer

    base_state = base_model.state_dict()
    skip_keys = {
        "embeddings.position_embeddings.weight",
        "embeddings.position_ids",
        "embeddings.token_type_ids",
        "pooler.dense.weight",
        "pooler.dense.bias",
    }
    loadable_state = {key: value for key, value in base_state.items() if key not in skip_keys}
    missing_keys, unexpected_keys = longformer_base.load_state_dict(loadable_state, strict=False)

    source_pos = base_model.embeddings.position_embeddings.weight.data
    target_pos = longformer_base.embeddings.position_embeddings.weight.data
    expanded_pos = _expand_position_embeddings(
        source_weight=source_pos,
        target_weight=target_pos,
        padding_idx=longformer_config.pad_token_id,
    )
    longformer_base.embeddings.position_embeddings.weight.data.copy_(expanded_pos)

    if hasattr(longformer_base.embeddings, "position_ids"):
        position_ids = torch.arange(longformer_config.max_position_embeddings).expand((1, -1))
        longformer_base.embeddings.position_ids.data = position_ids

    _copy_bert_mlm_head(source_model, longformer_model)

    for source_layer, longformer_layer in zip(
        base_model.encoder.layer,
        longformer_base.encoder.layer,
    ):
        longformer_layer.attention.self.query_global.load_state_dict(
            source_layer.attention.self.query.state_dict()
        )
        longformer_layer.attention.self.key_global.load_state_dict(
            source_layer.attention.self.key.state_dict()
        )
        longformer_layer.attention.self.value_global.load_state_dict(
            source_layer.attention.self.value.state_dict()
        )

    tokenizer.model_max_length = target_max_length
    tokenizer.init_kwargs["model_max_length"] = target_max_length

    longformer_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    summary = {
        "source_model": source_model_name,
        "output_dir": str(output_path),
        "model_type": "longformer",
        "target_max_length": int(target_max_length),
        "attention_window": int(attention_window),
        "pad_token_id": int(longformer_config.pad_token_id),
        "max_position_embeddings": int(longformer_config.max_position_embeddings),
        "missing_keys_after_base_copy": missing_keys,
        "unexpected_keys_after_base_copy": unexpected_keys,
    }
    _json_dump(output_path / "conversion_summary.json", summary)
    return summary


def freeze_longformer_for_mlm(model, freeze_mode: str) -> dict[str, int]:
    if freeze_mode == "none":
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        return {"total": total, "trainable": trainable}

    allowed_fragments = {"query_global", "key_global", "value_global", "position_embeddings"}
    if freeze_mode == "globals+positions+lm_head":
        allowed_fragments.add("lm_head")
    elif freeze_mode != "globals+positions":
        raise ValueError(
            "--freeze-mode must be one of: none, globals+positions, globals+positions+lm_head."
        )

    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        if any(fragment in name for fragment in allowed_fragments):
            parameter.requires_grad = True

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def build_global_attention_collator(tokenizer, mlm_probability: float):
    try:
        import torch
        from transformers import DataCollatorForLanguageModeling
    except ImportError as exc:
        raise ImportError(
            "Install `transformers` and `torch` to use MLM training."
        ) from exc

    class DataCollatorWithGlobalAttention(DataCollatorForLanguageModeling):
        def __call__(self, examples):
            batch = super().__call__(examples)
            input_ids = batch["input_ids"]
            global_attention_mask = (input_ids == self.tokenizer.mask_token_id).long()
            rows_without_mask = torch.where(global_attention_mask.sum(dim=1) == 0)[0]
            for row_idx in rows_without_mask.tolist():
                cls_positions = (input_ids[row_idx] == self.tokenizer.cls_token_id).nonzero(as_tuple=False)
                if cls_positions.numel() > 0:
                    global_attention_mask[row_idx, int(cls_positions[0].item())] = 1
                else:
                    global_attention_mask[row_idx, 0] = 1
            batch["global_attention_mask"] = global_attention_mask
            return batch

    return DataCollatorWithGlobalAttention(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )


def build_mixed_length_dataset(
    text_file_path: str | Path,
    tokenizer,
    max_length: int,
    char_window: int,
    short_sample_ratio: float,
    min_short_chars: int,
):
    try:
        from torch.utils.data import Dataset
    except ImportError as exc:
        raise ImportError("Install `torch` to use MLM training.") from exc

    class MixedLengthTextDataset(Dataset):
        def __init__(self):
            self.path = Path(text_file_path)
            self.text = self.path.read_text(encoding="utf-8")
            self.tokenizer = tokenizer
            self.max_length = int(max_length)
            self.char_window = int(char_window)
            self.short_sample_ratio = float(short_sample_ratio)
            self.min_short_chars = int(min_short_chars)

            if not self.text.strip():
                raise ValueError(f"The training corpus is empty: {self.path}")

        def __len__(self):
            return max(1, len(self.text) // max(1, self.char_window))

        def __getitem__(self, idx):
            del idx
            if random.random() < self.short_sample_ratio:
                sample_chars = random.randint(self.min_short_chars, self.char_window)
            else:
                sample_chars = self.char_window

            max_start = max(0, len(self.text) - sample_chars)
            start = random.randint(0, max_start) if max_start else 0
            chunk = self.text[start:start + sample_chars]
            return self.tokenizer(
                chunk,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )

    return MixedLengthTextDataset()


def run_mlm_training(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(args.seed)

    try:
        from transformers import (
            LongformerForMaskedLM,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "Install `transformers`, `torch` and `accelerate` to use MLM training."
        ) from exc

    if args.fp16 and args.bf16:
        raise ValueError("Use either --fp16 or --bf16, not both.")
    if not 0.0 <= args.short_sample_ratio <= 1.0:
        raise ValueError("--short-sample-ratio must be between 0 and 1.")
    if args.min_short_chars <= 0:
        raise ValueError("--min-short-chars must be positive.")
    if args.char_window < args.min_short_chars:
        raise ValueError("--char-window must be >= --min-short-chars.")

    tokenizer = _load_tokenizer(args.model_dir)
    tokenizer.model_max_length = args.max_length
    tokenizer.init_kwargs["model_max_length"] = args.max_length

    model = LongformerForMaskedLM.from_pretrained(args.model_dir)
    if getattr(model.config, "model_type", None) != "longformer":
        raise ValueError(
            f"{args.model_dir} is not a Longformer checkpoint. Run the `convert` command first."
        )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    freeze_stats = freeze_longformer_for_mlm(model, args.freeze_mode)

    train_dataset = build_mixed_length_dataset(
        text_file_path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        char_window=args.char_window,
        short_sample_ratio=args.short_sample_ratio,
        min_short_chars=args.min_short_chars,
    )
    eval_dataset = None
    if args.validation_file:
        eval_dataset = build_mixed_length_dataset(
            text_file_path=args.validation_file,
            tokenizer=tokenizer,
            max_length=args.max_length,
            char_window=args.char_window,
            short_sample_ratio=args.short_sample_ratio,
            min_short_chars=args.min_short_chars,
        )

    data_collator = build_global_attention_collator(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
    )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        overwrite_output_dir=args.overwrite_output_dir,
        do_eval=eval_dataset is not None,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        optim=args.optim,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        prediction_loss_only=True,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.num_workers,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    summary = {
        "model_dir": args.model_dir,
        "output_dir": str(output_path),
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file) if args.validation_file else None,
        "max_length": int(args.max_length),
        "char_window": int(args.char_window),
        "short_sample_ratio": float(args.short_sample_ratio),
        "mlm_probability": float(args.mlm_probability),
        "epochs": float(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "grad_accum": int(args.grad_accum),
        "learning_rate": float(args.learning_rate),
        "freeze_mode": args.freeze_mode,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "trainable_parameters": int(freeze_stats["trainable"]),
        "total_parameters": int(freeze_stats["total"]),
    }
    _json_dump(output_path / "mlm_training_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a BERT-style RuBERT checkpoint into a Longformer and optionally "
            "continue MLM pretraining on long texts."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a BERT-style MLM checkpoint (for example DeepPavlov/rubert-base-cased) to Longformer.",
    )
    convert_parser.add_argument(
        "--source-model",
        default="DeepPavlov/rubert-base-cased",
        help="Source Hugging Face checkpoint or local path.",
    )
    convert_parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to save the converted Longformer checkpoint.",
    )
    convert_parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Target sequence length after conversion.",
    )
    convert_parser.add_argument(
        "--attention-window",
        type=int,
        default=512,
        help="Sliding-window size for local attention. Must be even.",
    )
    convert_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face loaders.",
    )
    convert_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed.",
    )

    mlm_parser = subparsers.add_parser(
        "mlm",
        help="Continue MLM pretraining on long texts with global attention on [MASK] tokens.",
    )
    mlm_parser.add_argument(
        "--model-dir",
        required=True,
        help="Converted Longformer checkpoint directory.",
    )
    mlm_parser.add_argument(
        "--train-file",
        required=True,
        help="Plain-text corpus for MLM pretraining.",
    )
    mlm_parser.add_argument(
        "--validation-file",
        default=None,
        help="Optional plain-text validation corpus.",
    )
    mlm_parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to save the continued-pretraining checkpoint.",
    )
    mlm_parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Token length used for MLM training.",
    )
    mlm_parser.add_argument(
        "--char-window",
        type=int,
        default=20000,
        help="Approximate number of source characters sampled before tokenization.",
    )
    mlm_parser.add_argument(
        "--short-sample-ratio",
        type=float,
        default=0.333,
        help="Probability of sampling a shorter chunk instead of a full-length one.",
    )
    mlm_parser.add_argument(
        "--min-short-chars",
        type=int,
        default=2048,
        help="Minimum number of source characters for short samples.",
    )
    mlm_parser.add_argument(
        "--mlm-probability",
        type=float,
        default=0.15,
        help="Masking probability used by DataCollatorForLanguageModeling.",
    )
    mlm_parser.add_argument(
        "--epochs",
        type=float,
        default=2.0,
        help="Number of MLM epochs.",
    )
    mlm_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    mlm_parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Per-device eval batch size.",
    )
    mlm_parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    mlm_parser.add_argument(
        "--learning-rate",
        type=float,
        default=6e-4,
        help="Learning rate.",
    )
    mlm_parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.2,
        help="Warmup ratio.",
    )
    mlm_parser.add_argument(
        "--optim",
        default="adafactor",
        help="Optimizer passed to TrainingArguments.",
    )
    mlm_parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps.",
    )
    mlm_parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Run evaluation every N steps when --validation-file is provided.",
    )
    mlm_parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="How many checkpoints to keep.",
    )
    mlm_parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Trainer logging frequency.",
    )
    mlm_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    mlm_parser.add_argument(
        "--freeze-mode",
        default="globals+positions",
        choices=["none", "globals+positions", "globals+positions+lm_head"],
        help="Which weights stay trainable during MLM.",
    )
    mlm_parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    mlm_parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 training.",
    )
    mlm_parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 training.",
    )
    mlm_parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Pass overwrite_output_dir=True to TrainingArguments.",
    )
    mlm_parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume MLM training from a Trainer checkpoint directory.",
    )
    mlm_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "convert":
        summary = convert_rubert_to_longformer(
            source_model_name=args.source_model,
            output_dir=args.output_dir,
            target_max_length=args.max_length,
            attention_window=args.attention_window,
            trust_remote_code=args.trust_remote_code,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "mlm":
        summary = run_mlm_training(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
