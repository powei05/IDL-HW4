"""Response-only supervised fine-tuning for the P1 decoder."""
import json
import math
import yaml
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from .base_trainer import BaseTrainer
from ..decoding.sequence_generator import SequenceGenerator
from ..utils.webnlg import format_webnlg_prompt, score_webnlg


class SFTTrainer(BaseTrainer):
    """Train a decoder on prompt/response pairs.

    Complete the marked training, validation, and generation methods.
    Reuse your P1 model and decoding implementation.
    """
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        self.best_metric = -float("inf")  # Higher Kaggle score is better.
        self.amp_enabled = bool(config["training"].get("amp", True)) and torch.device(self.device).type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        # TODO: Use summed cross-entropy, ignoring labels equal to -100.
        self.criterion = nn.CrossEntropyLoss(
            ignore_index = -100,
            reduction = "sum"
        )  # TODO

    def _init_experiment(self, run_name, config_file):
        """Provided: use a new output directory, or explicitly resume an existing run."""
        from time import time_ns
        root = (
            Path(self.config.get("output_dir", "expts"))
            .expanduser()
            .resolve()
            / run_name
        )
        resume = self.config["training"].get("resume", False)
        if resume:
            last_checkpoint = root / "checkpoints" / "last.pth"
            if not last_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Cannot resume: {last_checkpoint} was not found. "
                    "If training stopped before the first checkpoint, set this run's resume flag to false and rerun."
                )
        else:
            if root.exists():
                # Preserve any experiment containing saved model weights.
                has_checkpoint = any(
                    path.is_file()
                    and path.suffix.lower() in {".pth", ".pt", ".ckpt"}
                    for path in root.rglob("*")
                )
                if has_checkpoint:
                    raise FileExistsError(
                        f"Experiment {run_name!r} already contains saved model weights. "
                        "Set resume to true to continue from last.pth, or choose a new run name."
                    )
                # Finish the notebook's previous W&B session before
                # moving its files.
                if self.use_wandb:
                    import wandb
                    if wandb.run is not None:
                        wandb.finish()
                archived_root = root.with_name(
                    f"{root.name}-before-first-checkpoint-{time_ns()}"
                )
                root.rename(archived_root)
                print(
                    "Note: The previous attempt has no saved checkpoint. "
                    f"Its files were preserved at {archived_root}. "
                    "Starting this experiment again from the P1 checkpoint."
                )
            root.mkdir(parents=True, exist_ok=False)
            (root / "config.yaml").write_text(yaml.safe_dump(self.config))
            (root / "model.txt").write_text(str(self.model) + "\n")
        checkpoint, attn, text = (root / part for part in ("checkpoints", "attn", "text"))
        for directory in (checkpoint, attn, text):
            directory.mkdir(exist_ok=True)
        if self.use_wandb:
            import wandb
            if wandb.run is not None:
                wandb.finish()
            self.wandb_run = wandb.init(
                project=self.config["training"].get("wandb_project", "hw4p2"),
                entity=self.config["training"].get("wandb_entity") or None,
                name=run_name, config=self.config,
                id=self.config["training"].get("wandb_run_id") or None,
                resume="must" if resume else None, dir=str(root))
        return root, checkpoint, attn, text, checkpoint / "best.pth", checkpoint / "last.pth"


    @staticmethod
    def _metrics(loss_sum, token_count):
        """Provided: perplexity here is response-token PPL, not character PPL."""
        if token_count <= 0:
            raise ValueError("No supervised response tokens")
        ce = loss_sum / token_count
        if not math.isfinite(ce):
            raise FloatingPointError("Non-finite response loss")
        return {"response_ce": ce, "response_ppl": math.exp(ce), "response_tokens": token_count}

    def _batch(self, batch):
        """Provided: move tensors and reject a batch without supervised targets."""
        inputs, labels = batch["input_ids"].to(self.device), batch["labels"].to(self.device)
        if inputs.ndim != 2 or inputs.shape != labels.shape:
            raise ValueError("input_ids and labels must have matching (B, T) shapes")
        count = int((labels != -100).sum())
        if count == 0:
            raise ValueError("Batch has no supervised response tokens")
        return inputs, labels, count

    def _train_epoch(self, dataloader):
        """Accumulate summed losses; normalize gradients by the cycle's token count."""

        self._bind_model_to_active_wandb()

        if self.optimizer is None or len(dataloader) == 0:
            raise ValueError("Optimizer is not initialized or dataloader is empty")

        accumulation = self.config["training"]["gradient_accumulation_steps"]
        if (
            not isinstance(accumulation, int)
            or isinstance(accumulation, bool)
            or accumulation <= 0
        ):
            raise ValueError("Invalid gradient_accumulation_steps")

        clip = self.config["training"]["gradient_clip"]
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("Invalid gradient_clip")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_tokens = 0
        cycle_tokens = 0

        bar = tqdm(dataloader)

        for index, batch in enumerate(bar):

            inputs, labels, count = self._prepare_batch(batch)

            with torch.autocast(
                device_type=torch.device(self.device).type,
                enabled=self.amp_enabled
            ):
                logits, _ = self.model(inputs)

                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")

            self.scaler.scale(loss).backward()

            total_loss += loss.item()
            total_tokens += count
            cycle_tokens += count

            reached_accumulation_boundary = (
                (index + 1) % accumulation == 0
            )

            reached_final_batch = (
                (index + 1) == len(dataloader)
            )

            if reached_accumulation_boundary or reached_final_batch:

                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)

                # Normalize by supervised token count
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(cycle_tokens)

                # Clip gradient norm
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=clip
                )

                # Apply optimizer step through gradient scaler
                old_scale = self.scaler.get_scale()

                self.scaler.step(self.optimizer)
                self.scaler.update()

                optimizer_step_ran = (
                    self.scaler.get_scale() >= old_scale
                )

                if optimizer_step_ran:
                    self.global_optimizer_step += 1

                    if self.scheduler is not None:
                        self.scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)
                cycle_tokens = 0

            bar.set_postfix(
                response_ce=total_loss / total_tokens
            )

        return self._metrics(total_loss, total_tokens), {}

        #raise NotImplementedError("Complete _train_epoch")

    @torch.inference_mode()
    def _validate_epoch(self, dataloader):
        """Compute response CE from total NLL / total supervised tokens."""
        # TODO: Evaluate without gradients; exclude prompt and padding targets.
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for batch in dataloader:

            inputs, labels, count = self._prepare_batch(batch)

            logits, _ = self.model(inputs)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

            total_loss += loss.item()
            total_tokens += count

        return self._metrics(total_loss, total_tokens)
        #raise NotImplementedError("Complete _validate_epoch")

    @torch.inference_mode()
    def generate(self, records):
        """Generate from the triples alone, using your P1 greedy decoder.

        Do not append references to prompts. Decode one unpadded prompt at a time.
        Limit total context and stop at generated EOS, excluding prompt text.
        """
        # TODO: Use SequenceGenerator and the model's score/score_cached methods.
        self.model.eval()

        predictions = []

        budget = self.config["generation"].get("max_new_tokens", 128)

        if (
            not isinstance(budget, int)
            or isinstance(budget, bool)
            or budget <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")

        for record in records:

            # Build prompt from RDF triples only
            prompt_ids = self.tokenizer.encode(
                format_webnlg_prompt(record["triples"])
            )

            if not prompt_ids or len(prompt_ids) >= self.model.max_len:
                raise ValueError("Invalid prompt length")

            # Convert prompt to tensor
            prompt = torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device=self.device
            )

            # Limit total generation length
            maximum_sequence_length = min(
                self.model.max_len,
                len(prompt_ids) + budget
            )

            # Use KV cache only for RoPE
            if self.model.use_rope:
                cached_score_function = self.model.score_cached
            else:
                cached_score_function = None

            # Initialize greedy decoder
            decoder = SequenceGenerator(
                score_fn=self.model.score,
                tokenizer=self.tokenizer,
                max_length=maximum_sequence_length,
                device=self.device,
                cache_score_fn=cached_score_function
            )

            # Generate response
            sequences, scores = decoder.generate_greedy(prompt)

            # Remove tokens after generated EOS
            sequence = decoder.post_process_sequence(
                sequences[0],
                self.tokenizer,
                start_index=len(prompt_ids)
            )

            # Remove prompt tokens
            generated_tokens = sequence[len(prompt_ids):]

            # Decode response only
            prediction = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            predictions.append(prediction)

        return predictions        
        #raise NotImplementedError("Complete generate")

    def evaluate(self, dataloader, records=None):
        """Provided: combine teacher-forced response loss with generated-text metrics."""
        metrics = self._validate_epoch(dataloader)
        if records is not None:
            predictions = self.generate(records)
            scores = score_webnlg(predictions, [row["references"] for row in records])
            metrics.update(bleu=scores["bleu"], rouge_l=scores["rouge_l"],
                           kaggle_score=scores["kaggle_score"], examples=len(records))
        return metrics

    def train(self, train_dataloader, val_dataloader, dev_records):
        """Run epochs and select the best generated dev Kaggle-score checkpoint.

        Save last.pth for resuming the next epoch and best.pth for prediction.
        Log token loss and generation scores as different quantities.
        """
        # TODO: Run training/evaluation, update best Kaggle score, log, and save checkpoints.
        if not dev_records:
            raise ValueError("dev_records must not be empty")

        for epoch in range(self.current_epoch, self.config["training"]["epochs"]):

            sampler = train_dataloader.sampler

            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            train_metrics, _ = self._train_epoch(train_dataloader)

            dev_metrics = self.evaluate(
                val_dataloader,
                dev_records
            )

            improved = dev_metrics["kaggle_score"] > self.best_metric

            if improved:
                self.best_metric = dev_metrics["kaggle_score"]

            # Log training and development results for epoch + 1
            # Use the existing trainer logging method here.

            self.current_epoch = epoch + 1

            if improved:
                # Save best.pth using the existing checkpoint method
                pass

        # Save last.pth and write history.json
        # Use the existing trainer methods here.
        raise NotImplementedError("Complete train")
