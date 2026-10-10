# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from trl.experimental.orpo import ORPOConfig, ORPOTrainer
from trl.experimental.orpo.orpo_trainer import log1mexp

from ..testing_utils import TrlTestCase, require_peft


class TestORPOTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # get t5 as seq2seq example:
        model_id = "trl-internal-testing/tiny-T5ForConditionalGeneration"
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype="float32")
        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)

    def test_trust_remote_code(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        with pytest.raises(ValueError, match="custom code"):
            ORPOTrainer(
                model=model_id,
                args=ORPOConfig(output_dir=self.tmp_dir, report_to="none"),
                processing_class=tokenizer,
                train_dataset=dataset,
            )

        trainer = ORPOTrainer(
            model=model_id,
            args=ORPOConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            processing_class=tokenizer,
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    @pytest.mark.parametrize(
        "name, config_name",
        [
            ("qwen", "standard_preference"),
            ("t5", "standard_implicit_prompt_preference"),
            ("qwen", "conversational_preference"),
        ],
    )
    def test_orpo_trainer(self, name, config_name):
        training_args = ORPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            learning_rate=9e-1,
            eval_strategy="steps",
            beta=0.1,
            report_to="none",
        )

        dataset = load_dataset("trl-internal-testing/zen", config_name)

        if name == "qwen":
            model = self.model
            tokenizer = self.tokenizer
        elif name == "t5":
            model = self.t5_model
            tokenizer = self.t5_tokenizer
            training_args.is_encoder_decoder = True

        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.equal(param, new_param)

    def test_orpo_trainer_truncation(self):
        # Each response is truncated to `max_length` minus the prompt length. The prompt is never truncated, so a prompt
        # longer than `max_length` is kept whole with an empty completion.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = ORPOConfig(output_dir=self.tmp_dir, report_to="none")
        full = ORPOTrainer(
            model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dataset
        ).train_dataset
        training_args = ORPOConfig(output_dir=self.tmp_dir, max_length=8, report_to="none")
        truncated = ORPOTrainer(
            model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dataset
        ).train_dataset

        for full_row, truncated_row in zip(full, truncated, strict=True):
            length = max(8, len(full_row["prompt_input_ids"]))
            assert truncated_row["chosen_input_ids"] == full_row["chosen_input_ids"][:length]
            assert truncated_row["rejected_input_ids"] == full_row["rejected_input_ids"][:length]

    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "dataset_dict",
            "none",
        ],
    )
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # ORPO tokenizes the eval dataset at init by calling `.map()` directly on it, so streaming (iterable) and
        # plain dict-of-datasets eval datasets are not yet supported; only `Dataset` and `DatasetDict` are tested.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            eval_split = load_dataset("trl-internal-testing/zen", "standard_preference", split="test")
            if eval_dataset_type == "dataset":
                eval_dataset = eval_split
            else:  # "dataset_dict"
                eval_dataset = DatasetDict({"data1": eval_split, "data2": eval_split})

        training_args = ORPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = ORPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
            # Each split was tokenized independently.
            assert "chosen_input_ids" in next(iter(trainer.eval_dataset["data1"]))
            assert "chosen_input_ids" in next(iter(trainer.eval_dataset["data2"]))
        else:
            assert "chosen_input_ids" in next(iter(trainer.eval_dataset))

    def test_orpo_trainer_processing_class_autoloaded(self):
        # processing_class is documented as optional: when omitted it should be
        # auto-loaded from the model, consistent with DPOTrainer / RewardTrainer.
        training_args = ORPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            beta=0.1,
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")
        trainer = ORPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
        )
        assert trainer.processing_class is not None
        trainer.train()
        assert trainer.state.log_history[-1]["train_loss"] is not None

    @require_peft
    def test_peft_init_is_seeded(self):
        from peft import LoraConfig

        # Two trainers with the same seed start from the same adapter weights
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        adapters = []
        for global_seed in range(2):
            torch.manual_seed(global_seed)  # a different global RNG state, as in two separate runs
            trainer = ORPOTrainer(
                model=self.model_id,
                args=ORPOConfig(output_dir=self.tmp_dir, report_to="none"),
                processing_class=self.tokenizer,
                train_dataset=dataset,
                peft_config=LoraConfig(),
            )
            adapters.append({n: p.clone() for n, p in trainer.model.named_parameters() if "lora_A" in n})

        assert adapters[0]
        for n, param in adapters[0].items():
            assert torch.equal(param, adapters[1][n]), f"Parameter {n} differs between the two trainers."

    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_preference",
            "standard_implicit_prompt_preference",
            "conversational_preference",
            "conversational_implicit_prompt_preference",
        ],
    )
    @require_peft
    def test_orpo_trainer_with_lora(self, config_name):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        training_args = ORPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            gradient_accumulation_steps=4,
            learning_rate=9e-1,
            eval_strategy="steps",
            beta=0.1,
            report_to="none",
        )

        dataset = load_dataset("trl-internal-testing/zen", config_name)

        trainer = ORPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            peft_config=lora_config,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            if "lora" in n:
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    assert not torch.equal(param, new_param)

    def test_compute_metrics(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        def dummy_compute_metrics(*args, **kwargs):
            return {"test": 0.0}

        training_args = ORPOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,
            per_device_train_batch_size=2,
            do_eval=True,
            eval_strategy="steps",
            eval_steps=1,
            per_device_eval_batch_size=2,
            report_to="none",
        )

        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=dummy_compute_metrics,
        )

        trainer.train()

        assert trainer.state.log_history[-2]["eval_test"] == 0.0


class TestLog1mexp(TrlTestCase):
    # Split point of `log1mexp`, -log(2)
    T = -0.6931471805599453

    def reference(self, x):
        """Float64 reference of `log1mexp`, free of the NaN-gradient issue under test."""
        x = x.to(torch.float64)
        return torch.where(x < self.T, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))

    def test_values_match_float64_reference(self):
        x = torch.tensor([-100.0, -20.0, -5.0, self.T, -0.1, -1e-3, -1e-6, -1e-8], dtype=torch.float32)
        torch.testing.assert_close(log1mexp(x), self.reference(x).to(torch.float32), rtol=1e-5, atol=1e-6)

    def test_gradient_near_zero_is_finite(self):
        # Regression test for https://github.com/huggingface/trl/issues/7608: for strictly negative fp32
        # log-probabilities close to zero, the forward value is finite but the gradient used to be NaN because
        # `torch.where` also backpropagates through the singular unselected `log1p(-exp(x))` branch.
        x = torch.tensor([-1e-8], dtype=torch.float32, requires_grad=True)
        log1mexp(x).backward()
        assert torch.isfinite(x.grad).all()
        x64 = x.detach().to(torch.float64)
        torch.testing.assert_close(x.grad.to(torch.float64), -torch.exp(x64) / (1 - torch.exp(x64)), rtol=1e-5, atol=0)

    def test_orpo_loss_gradient_is_finite_for_confident_model(self):
        # Averaged token log-probabilities of a confident model can reach the affected range while the loss stays
        # finite, so NaN parameter gradients would go unnoticed without this check.
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.eye(2))

        inputs = torch.zeros(2, 64, 2)
        inputs[..., 0] = 20
        inputs[0, 0, 0] = 14
        inputs[1, 0, 0] = 13.5
        logps = head(inputs).log_softmax(-1)[..., 0].mean(-1)
        chosen_logps, rejected_logps = logps.unbind()

        # Same odds-ratio term as `ORPOTrainer.odds_ratio_loss` for one preference pair.
        log_odds = (chosen_logps - rejected_logps) - (log1mexp(chosen_logps) - log1mexp(rejected_logps))
        loss = -chosen_logps - 0.1 * F.logsigmoid(log_odds)
        loss.backward()

        assert torch.isfinite(head.weight.grad).all()
