"""Synthesises a per-user text profile with a Qwen chat model and embeds it."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .config import LLMConfig
from .features import L1Summary, UserFeatures


def select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


SYSTEM_PROMPT = (
    "## Role\n"
    "You are a data-summarization expert proficient in information extraction "
    "and user-behavior analysis, skilled at summarizing user information and "
    "reasoning about user profiles.\n\n"
    "## Task\n"
    "Strictly follow the rules below to produce a high-fidelity summary of the "
    "input user information. Do not omit key information, do not fabricate "
    "content, and ensure traceability and completeness. The resulting profile "
    "will be embedded and clustered to form cross-activity semantic user "
    "groups, so emphasize stable, transferable long-term preferences and filter "
    "out transient noise, allowing semantically similar users to be grouped.\n\n"
    "## Input Format\n"
    "The input contains [Basic Info], [Transaction Behavior], and [Search "
    "Behavior]. Transaction behaviour is split into short-term (last 7 days), "
    "medium-term (last 30 days), and long-term (last 365 days) sections. Each "
    "transaction line has the format:\n"
    "  - L1 Category (Leaf Category - Purchase Count - Price Power)\n"
    "where Price Power is in [0, 1]: the closer to 1, the higher the price tier "
    "the user buys within that leaf category; 0.5 denotes the median tier.\n\n"
    "## Output Format\n"
    "[User Profile Summary]\n"
    "- Core Identity & Life Stage: ...\n"
    "- Interest Points: ...\n"
    "- Consumption Philosophy & Decision Drivers: ...\n\n"
    "## Module Specification\n"
    "Extract explicit information and infer implicit information to output a "
    "highly distinctive, low-ambiguity profile.\n"
    "1. Core Identity & Life Stage: Based on basic info and the life scenarios "
    "reflected by high-frequency consumption, precisely locate the user's life "
    "stage (e.g., parenting family, solo young adult) and primary social role "
    "(e.g., household purchasing decision-maker).\n"
    "2. Interest Points: Within a single paragraph, clearly integrate: "
    "(1) Primary: the 1-3 categories with the highest purchase counts and their "
    "consumption tier, representing core stable consumption; "
    "(2) Secondary: medium-to-low purchase-count categories and their tier, "
    "representing secondary consumption areas; "
    "(3) Recent: categories appearing in the short-term section but rare or "
    "absent in the long-term section, and their tier, reflecting potential new "
    "demand.\n"
    "3. Consumption Philosophy & Decision Drivers: Combining the core profile, "
    "overall price-power tendency, price-power differences across categories, "
    "and purchase patterns, summarize the user's core consumption values (e.g., "
    "quality-oriented, value-for-money first) and key decision factors (e.g., "
    "price sensitivity).\n\n"
    "## Notes\n"
    "1. The summary must stay faithful to the original information, concise yet "
    "complete.\n"
    "2. Summarize each module in a single paragraph with no internal "
    "subdivisions.\n"
    "3. Even when behavioral signals are sparse, still produce a usable profile "
    "by reasoning from whatever explicit and attribute cues are available."
)


def _render_transactions(summaries: List[L1Summary]) -> str:
    if not summaries:
        return "  (none)"
    lines: List[str] = []
    for s in summaries:
        for leaf, cnt, price in s.leaves:
            lines.append(
                f"  - {s.l1_category} ({leaf} - {cnt} - {price:.2f})")
    return "\n".join(lines)


def render_prompt(user: UserFeatures) -> str:
    age = "unknown" if user.age < 0 else user.age

    search_lines: List[str] = []
    if user.recent_searches:
        for s in user.recent_searches:
            search_lines.append(f"  - {s.ts:%Y-%m-%d} | {s.query}")
    else:
        search_lines.append("  (none)")

    blocks = [
        SYSTEM_PROMPT,
        "",
        "[Basic Info]",
        f"  age={age}, gender={user.gender}, city={user.city}",
        "",
        "[Transaction Behavior]",
        "Short-term (last 7 days):",
        _render_transactions(user.short_purchases),
        "Medium-term (last 30 days):",
        _render_transactions(user.medium_purchases),
        "Long-term (last 365 days):",
        _render_transactions(user.long_purchases),
        "",
        "[Search Behavior]",
        "\n".join(search_lines),
        "",
        "[User Profile Summary]",
    ]
    return "\n".join(blocks)


class LLMService:

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.device = select_device()
        self._chat_model = None
        self._chat_tokenizer = None
        self._embed_model = None
        self._embed_tokenizer = None

    def _load_chat(self) -> None:
        if self._chat_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._chat_tokenizer = AutoTokenizer.from_pretrained(self.cfg.chat_model)
        self._chat_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.chat_model,
            torch_dtype=torch.float32,
        ).to(self.device)
        self._chat_model.eval()

    def _load_embed(self) -> None:
        if self._embed_model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._embed_tokenizer = AutoTokenizer.from_pretrained(self.cfg.embed_model)
        self._embed_model = AutoModel.from_pretrained(
            self.cfg.embed_model,
            torch_dtype=torch.float32,
        ).to(self.device)
        self._embed_model.eval()

    def synthesize_profile(self, user: UserFeatures) -> str:
        prompt = render_prompt(user)
        self._load_chat()
        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        import re

        import torch

        messages = [{"role": "user", "content": prompt}]
        try:
            text = self._chat_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self._chat_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self._chat_tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self._chat_model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        out = self._chat_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
        return out

    def embed_profiles(self, profiles: Iterable[str]) -> np.ndarray:
        profiles = list(profiles)
        self._load_embed()
        vec = self._encode(profiles)
        truncated = vec[:, : self.cfg.truncated_dim]
        norms = np.linalg.norm(truncated, axis=1, keepdims=True) + 1e-9
        return (truncated / norms).astype(np.float32)

    def _encode(self, profiles: List[str]) -> np.ndarray:
        import torch

        outputs: List[np.ndarray] = []
        bs = 8
        for i in range(0, len(profiles), bs):
            batch = profiles[i:i + bs]
            inputs = self._embed_tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self._embed_model(**inputs)
            hidden = out.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).type_as(hidden)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            outputs.append(pooled.cpu().numpy())
        return np.concatenate(outputs, axis=0)
