"""The commentator: Qwen2-VL (LiveCC-7B-Base) with a response head (speak, p_t) and an ask head (q_t),
both reading the hidden state of the <|FLAG|> token that ends every second (Proact-VL style)."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration

ALPHA = 0.2          # weight of the response loss (paper)
GAMMA = 3.0          # weight of transition steps in the BCE [ours; not given in the paper]
HEAD_FILE = "response_head.pt"
ASK_FILE = "ask_head.pt"     # when to ask the research agent
ASK_POS_W = 8.0              # positive weight of the (rare) ask labels
ASK_DELTA = 2                # seconds between asking and the answer's arrival


class ResponseHead(nn.Module):
    """Minimal gated MLP -> scalar logit [ours: SwiGLU, hidden = d/4]."""

    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or d // 4
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.out = nn.Linear(hidden, 1)

    def forward(self, h):
        return self.out(F.silu(self.gate(h)) * self.up(h)).squeeze(-1)


def response_loss(logits, y, gamma=GAMMA):
    """Transition-weighted BCE + stability regularisation (paper §4.3). logits, y: (T,)."""
    p = torch.sigmoid(logits)
    w = torch.ones_like(y)
    if len(y) > 1:
        w[1:] = torch.where(y[1:] != y[:-1], torch.full_like(y[1:], gamma), torch.ones_like(y[1:]))
    cls = (w * F.binary_cross_entropy_with_logits(logits, y, reduction="none")).sum() / w.sum()
    reg = (p.mean() - y.mean()) ** 2
    if len(y) > 1:
        same = (y[1:] == y[:-1])
        if same.any():
            reg = reg + ((p[1:] - p[:-1]) ** 2)[same].mean()
    return cls + reg, {"cls": cls.detach(), "reg": reg.detach()}


class ProactCommentator(nn.Module):
    def __init__(self, base: Qwen2VLForConditionalGeneration, head: ResponseHead = None):
        super().__init__()
        self.base = base
        self.config = base.config
        d = base.config.hidden_size
        self.head = head or ResponseHead(d).to(dtype=base.dtype)
        self.ask_head = ResponseHead(d).to(dtype=base.dtype)
        self._penult = None
        # hidden state after the penultimate decoder layer (Proact-VL: "penultimate transformer layer")
        base.model.layers[-2].register_forward_hook(self._grab)
        self._final = None   # final (post-norm) hidden state, for token log-probs (streamduo.interact.ask_labels)
        base.model.norm.register_forward_hook(self._grab_final)

    def _grab(self, module, inp, out):
        self._penult = out[0] if isinstance(out, tuple) else out

    def _grab_final(self, module, inp, out):
        self._final = out

    def token_logprobs(self, input_ids, positions, b=0):
        """log p(input_ids[b, pos] | prefix) at the given positions, from the last forward pass."""
        h = self._final[b, positions - 1]
        logits = self.base.lm_head(h).float()
        return torch.log_softmax(logits, -1).gather(-1, input_ids[b, positions][:, None]).squeeze(-1)

    @classmethod
    def from_pretrained(cls, path, n_vocab=None, **kw):
        base = Qwen2VLForConditionalGeneration.from_pretrained(path, **kw)
        if n_vocab is not None and base.get_input_embeddings().weight.shape[0] < n_vocab:
            base.resize_token_embeddings(n_vocab)
        head = None
        hp = os.path.join(path, HEAD_FILE)
        if os.path.exists(hp):
            head = ResponseHead(base.config.hidden_size).to(dtype=base.dtype)
            head.load_state_dict(torch.load(hp, map_location="cpu"))
        model = cls(base, head)
        ap = os.path.join(path, ASK_FILE)
        if os.path.exists(ap):
            model.ask_head.load_state_dict(torch.load(ap, map_location="cpu"))
        return model

    def save_pretrained(self, path, **kw):
        self.base.save_pretrained(path, **kw)
        torch.save(self.head.state_dict(), os.path.join(path, HEAD_FILE))
        torch.save(self.ask_head.state_dict(), os.path.join(path, ASK_FILE))

    def gradient_checkpointing_enable(self, **kw):
        self.base.gradient_checkpointing_enable(**kw)

    def ask_logits(self, flag_index):
        return self.ask_head(self._penult[0, flag_index].to(self.ask_head.out.weight.dtype)).float()

    def flag_logits(self, flag_index):
        return self.head(self._penult[0, flag_index].to(self.head.out.weight.dtype)).float()

    def forward(self, input_ids, attention_mask=None, pixel_values_videos=None, video_grid_thw=None,
                labels=None, flag_index=None, flag_labels=None, ask_labels=None, **kw):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                        pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw,
                        labels=labels, use_cache=False)
        loss = out.loss
        stats = {}
        if flag_index is not None and flag_labels is not None:
            logits = self.flag_logits(flag_index)
            r_loss, stats = response_loss(logits, flag_labels.float())
            stats["resp"] = r_loss.detach()
            if loss is None or torch.isnan(loss):  # all-silent clip: no LM targets
                loss = ALPHA * r_loss
            else:
                stats["lm"] = loss.detach()
                loss = loss + ALPHA * r_loss
        if flag_index is not None and ask_labels is not None:
            a_logits = self.ask_logits(flag_index)
            a_loss = F.binary_cross_entropy_with_logits(
                a_logits, ask_labels.float(), pos_weight=torch.tensor(ASK_POS_W, device=a_logits.device))
            stats["ask"] = a_loss.detach()
            loss = ALPHA * a_loss if loss is None else loss + ALPHA * a_loss
        out.loss = loss
        out.stats = stats
        return out
