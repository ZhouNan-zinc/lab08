# -*- coding: utf-8 -*-
"""
实验八：自制语言模型的知识蒸馏（纯 CPU）

单文件实现：
  * 可复现的合成「仿古诗句」字符语料（训练集/留出集不同种子，留出集不含固定唐诗）
  * 字符级 Pre-Norm GPT（教师 C=160，学生 C=64，2 层 4 头，block=96）
  * 教师监督训练、学生从零训练（hard label）、蒸馏训练（soft label）
  * T x alpha 4x4 消融、FitNet 特征蒸馏、同结构自蒸馏、动态量化、教师规模扫描
  * 训练曲线、消融折线图、4x4 热力图

常用命令（在 lab08/ 下执行）：
  .venv/bin/python distill_lm.py --mode quick
  .venv/bin/python distill_lm.py --mode teacher --retrain
  .venv/bin/python distill_lm.py --mode all
  .venv/bin/python distill_lm.py --mode ablate
  .venv/bin/python distill_lm.py --mode feat --feat 0.1
  .venv/bin/python distill_lm.py --mode quant
  .venv/bin/python distill_lm.py --mode selfdistill
  .venv/bin/python distill_lm.py --mode scan
"""
import argparse
import json
import math
import os
import platform
import random
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEFAULT_DIRS = {
    "quick": RESULTS / "smoke" / "quick",
    "teacher": RESULTS / "formal",
    "all": RESULTS / "formal",
    "ablate": RESULTS / "ablate",
    "feat": RESULTS / "ext_feat",
    "quant": RESULTS / "ext_quant",
    "selfdistill": RESULTS / "ext_self",
    "scan": RESULTS / "ext_scan",
}

# ============================================================
# 1. 语料：可复现的合成「仿古诗句」
# ============================================================
SUBJ = ["小明", "小红", "阿远", "书生", "渔夫", "牧童", "旅人", "诗人",
        "少年", "故人", "归客", "山僧", "孤客", "游子"]
PLACE = ["江边", "山寺", "长亭", "渡口", "竹林", "荒村", "城楼", "野渡",
         "溪桥", "松间", "篱下", "窗前", "陌上", "舟中"]
VERB = ["望见", "听见", "想起", "送去", "寻到", "梦见", "独对", "遥看",
        "静听", "又逢", "恰遇", "空忆"]
OBJ = ["明月", "落花", "孤舟", "归雁", "残灯", "细雨", "青山", "晚钟",
       "白雪", "寒江", "疏星", "暮云", "秋草", "远帆"]
MOOD = ["心中惆怅", "不觉天明", "欣然归去", "泪湿衣襟", "长叹一声",
        "默然无语", "忽忆旧游", "欲语还休", "此意难平", "一笑而别"]
AUX = ["看", "听", "守", "忆"]
ADV = ["正好", "可怜", "可惜", "依旧"]

POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",
]


def build_corpus_lines(n_lines=3000, seed=0, include_poems=True):
    """生成合成句列表；留出集换种子，并可排除固定唐诗以避免与训练集重复。"""
    rng = random.Random(seed)
    lines = []
    for i in range(n_lines):
        s, p, v, o, m = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            lines.append("%s%s%s%s，%s。" % (s, p, v, o, m))
        elif i % 3 == 1:
            lines.append("%s 在%s%s 了一夜%s，%s。" %
                         (s, p, o, rng.choice(AUX), m))
        else:
            lines.append("%s%s 道：“%s%s。”%s。" %
                         (s, v, o, rng.choice(ADV), m))
    if include_poems:
        lines.extend(POEMS)
    return lines


def build_corpus(n_lines=3000, seed=0, include_poems=True):
    return "".join(build_corpus_lines(n_lines, seed, include_poems))


class Tokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def make_data(block_size):
    train_lines = build_corpus_lines(3000, seed=0, include_poems=True)
    val_lines = build_corpus_lines(300, seed=999, include_poems=False)
    train_set = set(train_lines)
    removed = [ln for ln in val_lines if ln in train_set]
    val_lines = [ln for ln in val_lines if ln not in train_set]
    train_text = "".join(train_lines)
    val_text = "".join(val_lines)
    tok = Tokenizer(train_text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    meta = {
        "vocab": tok.vocab_size,
        "train_chars": int(len(train_ids)),
        "val_chars": int(len(val_ids)),
        "train_lines": len(train_lines),
        "val_lines": len(val_lines),
        "val_lines_removed_as_duplicates": len(removed),
        "block": block_size,
        "val_contains_fixed_poems": False,
    }
    return tok, train_ids, val_ids, meta


# ============================================================
# 2. 字符级 GPT
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        assert C % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.c_attn = nn.Linear(C, 3 * C)
        self.c_proj = nn.Linear(C, C)
        self.drop = nn.Dropout(cfg["dropout"])
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
            .view(1, 1, cfg["block_size"], cfg["block_size"]))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        self.fc = nn.Linear(C, 4 * C)
        self.proj = nn.Linear(4 * C, C)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], vocab_size, bias=False)
        self.apply(self._init)
        self.n_param = sum(p.numel() for p in self.parameters())

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, need_hidden=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
        if need_hidden:
            return logits, loss, x
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ============================================================
# 3. 批采样与评估
# ============================================================
def get_batch(data, bs, block, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data, bs, block, iters=20, seed=1234):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, bs, block, gen)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


@torch.no_grad()
def agreement(student, teacher, data, bs, block, iters=10, seed=4321):
    student.eval()
    teacher.eval()
    gen = torch.Generator().manual_seed(seed)
    hit = tot = 0
    for _ in range(iters):
        x, _ = get_batch(data, bs, block, gen)
        a = student(x)[0].argmax(-1)
        b = teacher(x)[0].argmax(-1)
        hit += (a == b).sum().item()
        tot += b.numel()
    student.train()
    teacher.train()
    return hit / tot


# ============================================================
# 4. 训练：hard label / soft label（KD）/ FitNet
# ============================================================
def train(model, data, iters, bs, block, lr=3e-3, seed=0,
          teacher=None, T=4.0, alpha=0.5, feat_weight=0.0,
          log_every=50, tag="", record_parts=False):
    """
    alpha = 0   -> 纯交叉熵（从零训练对照组）
    alpha > 0   -> L = (1-alpha)*CE + alpha*T^2*KL(teacher || student)
    feat_weight -> 额外加最后一层隐状态 MSE（FitNet 式特征蒸馏）
    """
    proj = None
    if teacher is not None and feat_weight > 0:
        proj = nn.Linear(model.cfg["n_embd"], teacher.cfg["n_embd"])
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 1)
    hist, hist_ce, hist_kd = [], [], []
    if teacher is not None:
        teacher.eval()
    model.train()
    t0 = time.time()
    for it in range(iters):
        x, y = get_batch(data, bs, block, gen)
        if proj is not None:
            logits, ce, hidden = model(x, y, need_hidden=True)
        else:
            logits, ce = model(x, y)
            hidden = None
        loss = ce
        kd_val = None
        if teacher is not None and alpha > 0:
            with torch.no_grad():
                t_logits, _, t_hidden = teacher(x, need_hidden=True)
            # 关键：先 reshape 成 (B*T, V) 再算 KL。直接对 (B,T,V) 用
            # reduction="batchmean" 只会除以 B，序列维被重复累加，KL 会被放大 T 倍。
            V = logits.size(-1)
            s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
            t_p = F.softmax(t_logits.reshape(-1, V) / T, dim=-1)
            kd_val = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
            loss = (1 - alpha) * ce + alpha * kd_val
            if proj is not None:
                loss = loss + feat_weight * F.mse_loss(proj(hidden), t_hidden)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(loss.item())
        if record_parts:
            hist_ce.append(float(ce.item()))
            hist_kd.append(float(kd_val.item()) if kd_val is not None else None)
        if (it + 1) % log_every == 0 or it == 0:
            extra = ""
            if kd_val is not None:
                extra = " ce %.4f kd(T^2) %.4f" % (ce.item(), kd_val.item())
            print(" [%s] iter %4d/%d loss %.4f%s (%.1fs)"
                  % (tag or "train", it + 1, iters, loss.item(), extra,
                     time.time() - t0), flush=True)
    out = {"loss": hist, "time_s": time.time() - t0}
    if record_parts:
        out["ce"] = hist_ce
        out["kd"] = hist_kd
    return out


# ============================================================
# 5. 消融、绘图、量化
# ============================================================
def ablation(args, tok, train_ids, teacher, val_ids, bs, block, out_dir):
    grid_T = [1, 2, 4, 8]
    grid_a = [0.3, 0.5, 0.7, 0.9]
    V = tok.vocab_size
    gen = torch.Generator().manual_seed(2024)
    cache = []
    teacher.eval()
    with torch.no_grad():
        for _ in range(32):
            x, y = get_batch(train_ids, bs, block, gen)
            cache.append((x, y, teacher(x)[0]))
    rows = []
    t_all = time.time()
    for T in grid_T:
        for a in grid_a:
            torch.manual_seed(args.seed + 7)
            m = GPT(tok.vocab_size, args.student_cfg)
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
            m.train()
            t0 = time.time()
            for it in range(args.abl_iters):
                x, y, tl = cache[it % len(cache)]
                logits = m(x)[0]
                ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
                s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
                t_p = F.softmax(tl.reshape(-1, V) / T, dim=-1)
                kd = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
                loss = (1 - a) * ce + a * kd
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            v = evaluate(m, val_ids, bs, block)
            rows.append({"T": T, "alpha": a, "val_loss": v,
                         "ppl": math.exp(v), "time_s": time.time() - t0})
            print(" T=%d alpha=%.1f -> val %.4f (%.1fs)"
                  % (T, a, v, time.time() - t0), flush=True)
    best = min(rows, key=lambda r: r["val_loss"])
    worst = max(rows, key=lambda r: r["val_loss"])
    print("[ablate] 最优 T=%d alpha=%.1f val=%.4f ｜ 最差 T=%d alpha=%.1f val=%.4f"
          % (best["T"], best["alpha"], best["val_loss"],
             worst["T"], worst["alpha"], worst["val_loss"]), flush=True)
    return {"rows": rows, "best": best, "worst": worst,
            "total_time_s": time.time() - t_all, "cache_batches": len(cache),
            "abl_iters": args.abl_iters}


def plot_curves(res, path, title="training loss"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[plot] skip (%s)" % e)
        return
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, lab in (("teacher", "Teacher"), ("scratch", "Student-scratch"),
                   ("kd", "Student-KD")):
        if k in res and res[k].get("curve"):
            c = res[k]["curve"]
            ax[0].plot(range(1, len(c) + 1), c, label=lab, linewidth=1.2)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("training loss")
    ax[0].set_title(title)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    if "scratch" in res and "kd" in res:
        names = ["scratch", "kd"]
        vals = [res["scratch"]["val_loss"], res["kd"]["val_loss"]]
        bars = ax[1].bar(names, vals, color=["#6b7280", "#2563eb"])
        ax[1].set_ylabel("held-out loss")
        ax[1].set_title("held-out loss")
        ax[1].grid(alpha=0.3, axis="y")
        for b, v in zip(bars, vals):
            ax[1].text(b.get_x() + b.get_width() / 2, v, "%.4f" % v,
                       ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("[plot] saved %s" % path, flush=True)


def plot_ablation_lines(abl, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[plot] skip (%s)" % e)
        return
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rows = abl["rows"]
    for T in sorted({r["T"] for r in rows}):
        xs = [r["alpha"] for r in rows if r["T"] == T]
        ys = [r["val_loss"] for r in rows if r["T"] == T]
        ax.plot(xs, ys, marker="o", linewidth=1.2, label="T=%d" % T)
    ax.set_xlabel("alpha")
    ax.set_ylabel("held-out loss")
    ax.set_title("T / alpha ablation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("[plot] saved %s" % path, flush=True)


def plot_heatmap(abl, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print("[plot] skip (%s)" % e)
        return
    plt.rcParams["axes.unicode_minus"] = False
    Ts = sorted({r["T"] for r in abl["rows"]})
    As = sorted({r["alpha"] for r in abl["rows"]})
    M = np.full((len(Ts), len(As)), float("nan"))
    for r in abl["rows"]:
        M[Ts.index(r["T"]), As.index(r["alpha"])] = r["val_loss"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(As)))
    ax.set_xticklabels(["%.1f" % a for a in As])
    ax.set_yticks(range(len(Ts)))
    ax.set_yticklabels(["%d" % t for t in Ts])
    ax.set_xlabel("alpha")
    ax.set_ylabel("temperature T")
    ax.set_title("held-out loss (T x alpha)")
    for i in range(len(Ts)):
        for j in range(len(As)):
            v = M[i, j]
            ax.text(j, i, "%.4f" % v, ha="center", va="center",
                    color="white" if v > (np.nanmin(M) + np.nanmax(M)) / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="held-out loss")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("[plot] saved %s" % path, flush=True)


def model_file_size(model, path):
    torch.save(model.state_dict(), path)
    return os.path.getsize(path)


@torch.no_grad()
def forward_latency(model, bs, block, repeats=10, warmup=3):
    model.eval()
    x = torch.randint(0, 10, (bs, block))
    for _ in range(warmup):
        model(x)
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(x)
    dt = (time.perf_counter() - t0) / repeats
    model.train()
    return dt * 1000.0


# ============================================================
# 6. 运行工具
# ============================================================
def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def env_snapshot():
    cpu = ""
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        cpu = platform.processor()
    mem = ""
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemTotal"):
                mem = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    import matplotlib
    return {
        "platform": platform.platform(),
        "os": platform.system() + " " + platform.release(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "memory": mem,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "time": datetime.now().isoformat(timespec="seconds"),
    }


def analytical_params(vocab, cfg):
    C, L, T = cfg["n_embd"], cfg["n_layer"], cfg["block_size"]
    return vocab * C + T * C + L * (12 * C * C + 13 * C) + 2 * C + vocab * C


def self_test(tok, train_ids, val_ids, args):
    checks = []
    # 数据 shape / dtype / 标签范围
    gen = torch.Generator().manual_seed(args.seed + 1)
    x, y = get_batch(train_ids, args.batch_size, args.block_size, gen)
    checks.append(("batch_shape", list(x.shape) == [args.batch_size, args.block_size]
                   and list(y.shape) == [args.batch_size, args.block_size]))
    checks.append(("batch_dtype", x.dtype == torch.long and y.dtype == torch.long))
    checks.append(("label_range", int(y.min()) >= 0 and int(y.max()) < tok.vocab_size))
    # scratch / kd 首批数据完全一致
    g1 = torch.Generator().manual_seed(args.seed + 1)
    x1, y1 = get_batch(train_ids, args.batch_size, args.block_size, g1)
    g2 = torch.Generator().manual_seed(args.seed + 1)
    x2, y2 = get_batch(train_ids, args.batch_size, args.block_size, g2)
    checks.append(("same_data_order", torch.equal(x1, x2) and torch.equal(y1, y2)))
    # 同一初始化种子两次初始化一致
    torch.manual_seed(args.seed + 7)
    a = GPT(tok.vocab_size, args.student_cfg)
    torch.manual_seed(args.seed + 7)
    b = GPT(tok.vocab_size, args.student_cfg)
    checks.append(("same_init", all(torch.equal(p, q) for p, q in
                                    zip(a.state_dict().values(), b.state_dict().values()))))
    # 参数量公式
    pa = analytical_params(tok.vocab_size, args.student_cfg)
    err = abs(pa - a.n_param) / a.n_param
    checks.append(("param_formula_error_lt_1pct", err < 0.01))
    # 因果性：改变最后一个输入不应改变前面位置的输出
    a.eval()
    with torch.no_grad():
        z = torch.zeros(1, args.block_size, dtype=torch.long)
        l1 = a(z)[0]
        z2 = z.clone()
        z2[0, -1] = (z2[0, -1] + 1) % tok.vocab_size
        l2 = a(z2)[0]
    checks.append(("causal_mask", torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-6)))
    # 损失有限
    _, loss = a(z, z)
    checks.append(("loss_finite", math.isfinite(float(loss.detach()))))
    details = {"checks": [{"name": n, "pass": bool(ok)} for n, ok in checks],
               "student_param_analytical": pa,
               "student_param_actual": a.n_param}
    return details


class RunLogger:
    def __init__(self, mode, out_dir, args):
        self.meta = {
            "mode": mode,
            "command": " ".join([Path(__file__).name] + os.sys.argv[1:]),
            "argv": os.sys.argv,
            "out_dir": str(out_dir),
            "config": vars(args),
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
        }
        self.path = out_dir / "run_meta.json"

    def finish(self, status="success", error=None):
        self.meta["end_time"] = datetime.now().isoformat(timespec="seconds")
        self.meta["status"] = status
        if error:
            self.meta["error"] = error
        t0 = datetime.fromisoformat(self.meta["start_time"])
        t1 = datetime.fromisoformat(self.meta["end_time"])
        self.meta["duration_s"] = (t1 - t0).total_seconds()
        save_json(self.meta, self.path)


def make_cfg(n_layer, n_head, n_embd, block_size, dropout=0.0):
    return dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                block_size=block_size, dropout=dropout)


def new_model(vocab, cfg, seed):
    torch.manual_seed(seed)
    return GPT(vocab, cfg)


def load_or_train_teacher(args, tok, train_ids, val_ids, bs, block, out_dir, cfg=None,
                          allow_formal_fallback=True):
    cfg = cfg or args.teacher_cfg
    if args.teacher_path:
        path = Path(args.teacher_path)
    elif (out_dir / "teacher.pt").exists():
        path = out_dir / "teacher.pt"
    elif allow_formal_fallback and (RESULTS / "formal" / "teacher.pt").exists():
        path = RESULTS / "formal" / "teacher.pt"
    else:
        path = out_dir / "teacher.pt"
    teacher = new_model(tok.vocab_size, cfg, args.seed)
    print("[teacher] params = %.3f M" % (teacher.n_param / 1e6), flush=True)
    row = {"params": teacher.n_param, "cfg": cfg}
    if path.exists() and not args.retrain:
        teacher.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        tv = evaluate(teacher, val_ids, bs, block)
        row.update({"val_loss": tv, "ppl": math.exp(tv), "time_s": 0.0,
                    "curve": [], "cached": True, "path": str(path)})
        # 若同目录保存过该教师的训练结果，补齐训练曲线与真实耗时，
        # 供跨模式绘图/汇总使用（不改变权重，只复用同一教师记录）。
        tr_path = path.parent / "results_teacher.json"
        if tr_path.exists():
            try:
                tr = load_json(tr_path)["teacher"]
                if tr.get("params") == teacher.n_param:
                    row["curve"] = tr.get("curve", [])
                    row["time_s"] = tr.get("time_s", 0.0)
            except Exception:
                pass
        print("[teacher] reuse %s val_loss=%.4f ppl=%.2f"
              % (path.name, tv, math.exp(tv)), flush=True)
    else:
        hist = train(teacher, train_ids, args.teacher_iters, bs, block,
                     lr=args.lr, seed=args.seed, log_every=args.log_every,
                     tag="teacher", record_parts=False)
        tv = evaluate(teacher, val_ids, bs, block)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(teacher.state_dict(), path)
        row.update({"val_loss": tv, "ppl": math.exp(tv),
                    "time_s": hist["time_s"], "curve": hist["loss"],
                    "cached": False, "path": str(path)})
        print("[teacher] val_loss=%.4f ppl=%.2f %.1fs"
              % (tv, math.exp(tv), hist["time_s"]), flush=True)
    return teacher, row


def eval_student(m, hist, name, teacher, val_ids, bs, block, extra=None):
    v = evaluate(m, val_ids, bs, block)
    row = {"params": m.n_param, "val_loss": v, "ppl": math.exp(v),
           "time_s": hist["time_s"], "curve": hist["loss"],
           "agree": agreement(m, teacher, val_ids, bs, block) if teacher is not None else None,
           "cfg": m.cfg}
    if hist.get("ce") is not None:
        row["curve_ce"] = hist["ce"]
    if hist.get("kd") is not None:
        row["curve_kd"] = hist["kd"]
    if extra:
        row.update(extra)
    print("[%s] val_loss=%.4f ppl=%.2f agree=%s %.1fs"
          % (name, v, math.exp(v),
             "%.3f" % row["agree"] if row["agree"] is not None else "—",
             hist["time_s"]), flush=True)
    return row


def generate_all(tok, teacher, s0, s1):
    prompt = "月"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = {}
    for name, model in (("teacher", teacher), ("scratch", s0), ("kd", s1)):
        torch.manual_seed(2026)
        with torch.no_grad():
            g = model.generate(ids, 60, temperature=0.9, top_k=20)
        text = tok.decode(g[0].tolist())
        out[name] = text
        print("[generate-%s] %s" % (name, text.replace("\n", " ")), flush=True)
    return {"prompt": prompt, "temperature": 0.9, "top_k": 20, "samples": out}


# ============================================================
# 7. 各运行模式
# ============================================================
def run_teacher(args, out_dir):
    tok, train_ids, val_ids, data_meta = make_data(args.block_size)
    print("[data] %s" % data_meta, flush=True)
    teacher, row = load_or_train_teacher(args, tok, train_ids, val_ids,
                                         args.batch_size, args.block_size, out_dir)
    res = {"data": data_meta, "teacher": row,
           "env": env_snapshot()}
    save_json(res, out_dir / "results_teacher.json")
    return res


def run_all(args, out_dir):
    tok, train_ids, val_ids, data_meta = make_data(args.block_size)
    print("[data] %s" % data_meta, flush=True)
    checks = self_test(tok, train_ids, val_ids, args)
    save_json(checks, out_dir / "self_test.json")
    print("[self-test] %s" % ("PASS" if all(c["pass"] for c in checks["checks"]) else "FAIL"),
          flush=True)
    if not all(c["pass"] for c in checks["checks"]):
        raise RuntimeError("self-test failed: %s" % checks)
    teacher, trow = load_or_train_teacher(args, tok, train_ids, val_ids,
                                          args.batch_size, args.block_size, out_dir,
                                          allow_formal_fallback=(args.mode != "quick"))
    bs, block = args.batch_size, args.block_size
    probe = new_model(tok.vocab_size, args.student_cfg, args.seed + 7)
    sp = probe.n_param
    print("[student] params = %.3f M (压缩率 %.2fx)"
          % (sp / 1e6, teacher.n_param / sp), flush=True)

    s0 = new_model(tok.vocab_size, args.student_cfg, args.seed + 7)
    h0 = train(s0, train_ids, args.student_iters, bs, block, lr=args.lr,
               seed=args.seed, log_every=args.log_every, tag="scratch",
               record_parts=True)
    row0 = eval_student(s0, h0, "scratch", teacher, val_ids, bs, block,
                        {"teacher": None, "T": None, "alpha": 0.0, "feat": 0.0})
    torch.save(s0.state_dict(), out_dir / "student_scratch.pt")

    s1 = new_model(tok.vocab_size, args.student_cfg, args.seed + 7)
    h1 = train(s1, train_ids, args.student_iters, bs, block, lr=args.lr,
               seed=args.seed, teacher=teacher, T=args.T, alpha=args.alpha,
               feat_weight=args.feat, log_every=args.log_every, tag="kd",
               record_parts=True)
    row1 = eval_student(s1, h1, "kd", teacher, val_ids, bs, block,
                        {"teacher": "teacher", "T": args.T, "alpha": args.alpha,
                         "feat": args.feat})
    torch.save(s1.state_dict(), out_dir / "student_kd.pt")

    rel = 100 * (row0["val_loss"] - row1["val_loss"]) / row0["val_loss"]
    print("\n[对比] 从零 %.4f -> 蒸馏 %.4f  相对下降 %.2f%%"
          % (row0["val_loss"], row1["val_loss"], rel), flush=True)

    samples = generate_all(tok, teacher, s0, s1)
    res = {"data": data_meta, "self_test": checks, "teacher": trow,
           "scratch": row0, "kd": row1, "comparison": {
               "relative_drop_pct": rel,
               "loss_delta": row1["val_loss"] - row0["val_loss"],
               "ppl_delta": row1["ppl"] - row0["ppl"],
               "agree_delta": row1["agree"] - row0["agree"]},
           "samples": samples, "env": env_snapshot()}
    save_json(res, out_dir / "results_all.json")
    plot_curves(res, out_dir / "curves.png")
    # 正式运行的图复制到公共图目录，供报告引用；quick 不污染正式图
    if args.mode == "all":
        figs = RESULTS / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_dir / "curves.png", figs / "curves.png")
    return res


def run_ablate(args, out_dir):
    tok, train_ids, val_ids, data_meta = make_data(args.block_size)
    print("[data] %s" % data_meta, flush=True)
    teacher, trow = load_or_train_teacher(args, tok, train_ids, val_ids,
                                          args.batch_size, args.block_size,
                                          out_dir)
    abl = ablation(args, tok, train_ids, teacher, val_ids,
                   args.batch_size, args.block_size, out_dir)
    res = {"data": data_meta, "teacher": trow, "ablation": abl,
           "env": env_snapshot()}
    save_json(res, out_dir / "results_ablate.json")
    plot_ablation_lines(abl, out_dir / "ablation_curves.png")
    plot_heatmap(abl, out_dir / "heatmap.png")
    figs = RESULTS / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    if args.abl_iters == 80:
        # 正式 80 步消融对应报告表 7 / 图 2
        shutil.copy2(out_dir / "ablation_curves.png", figs / "ablation_curves.png")
        shutil.copy2(out_dir / "heatmap.png", figs / "heatmap.png")
    else:
        # 补充的 120 步复测单独命名，避免覆盖正式图
        shutil.copy2(out_dir / "ablation_curves.png",
                     figs / ("ablation_curves_%dsteps.png" % args.abl_iters))
        shutil.copy2(out_dir / "heatmap.png",
                     figs / ("heatmap_%dsteps.png" % args.abl_iters))
    return res


def run_feat(args, out_dir):
    tok, train_ids, val_ids, data_meta = make_data(args.block_size)
    print("[data] %s" % data_meta, flush=True)
    teacher, trow = load_or_train_teacher(args, tok, train_ids, val_ids,
                                          args.batch_size, args.block_size,
                                          out_dir)
    bs, block = args.batch_size, args.block_size
    s0 = new_model(tok.vocab_size, args.student_cfg, args.seed + 7)
    h0 = train(s0, train_ids, args.student_iters, bs, block, lr=args.lr,
               seed=args.seed, log_every=args.log_every, tag="scratch",
               record_parts=True)
    row0 = eval_student(s0, h0, "scratch", teacher, val_ids, bs, block,
                        {"T": None, "alpha": 0.0, "feat": 0.0})
    s1 = new_model(tok.vocab_size, args.student_cfg, args.seed + 7)
    h1 = train(s1, train_ids, args.student_iters, bs, block, lr=args.lr,
               seed=args.seed, teacher=teacher, T=args.T, alpha=args.alpha,
               feat_weight=args.feat, log_every=args.log_every, tag="kd+feat",
               record_parts=True)
    row1 = eval_student(s1, h1, "kd_feat", teacher, val_ids, bs, block,
                        {"T": args.T, "alpha": args.alpha, "feat": args.feat})
    torch.save(s1.state_dict(), out_dir / "student_feat.pt")
    res = {"data": data_meta, "teacher": trow, "scratch": row0,
           "kd_feat": row1, "env": env_snapshot()}
    save_json(res, out_dir / "results_feat.json")
    return res


def run_quant(args, out_dir):
    tok, train_ids, val_ids, data_meta = make_data(args.block_size)
    print("[data] %s" % data_meta, flush=True)
    student_path = Path(args.student_path) if args.student_path else RESULTS / "formal" / "student_kd.pt"
    if not student_path.exists():
        raise FileNotFoundError("找不到蒸馏学生权重：%s，请先运行 --mode all" % student_path)
    cfg_path = student_path.parent / "results_all.json"
    res_all = json.load(open(cfg_path, encoding="utf-8")) if cfg_path.exists() else {}
    cfg = res_all.get("kd", {}).get("cfg", args.student_cfg)
    fp32 = GPT(tok.vocab_size, cfg)
    fp32.load_state_dict(torch.load(student_path, map_location="cpu", weights_only=True))
    fp32.eval()
    v_fp32 = evaluate(fp32, val_ids, args.batch_size, args.block_size)
    size_fp32 = model_file_size(fp32, out_dir / "student_fp32.pt")
    lat_fp32 = forward_latency(fp32, args.batch_size, args.block_size)

    q = torch.ao.quantization.quantize_dynamic(
        fp32, {nn.Linear}, dtype=torch.qint8)
    q.eval()
    v_int8 = evaluate(q, val_ids, args.batch_size, args.block_size)
    size_int8 = model_file_size(q, out_dir / "student_int8.pt")
    lat_int8 = forward_latency(q, args.batch_size, args.block_size)
    res = {"data": data_meta, "student_path": str(student_path),
           "fp32": {"val_loss": v_fp32, "ppl": math.exp(v_fp32),
                    "file_size_bytes": size_fp32,
                    "latency_ms": lat_fp32},
           "int8": {"val_loss": v_int8, "ppl": math.exp(v_int8),
                    "file_size_bytes": size_int8,
                    "latency_ms": lat_int8},
           "size_ratio_int8_over_fp32": size_int8 / size_fp32,
           "latency_ratio_int8_over_fp32": lat_int8 / lat_fp32,
           "env": env_snapshot()}
    save_json(res, out_dir / "results_quant.json")
    print("[quant] fp32 %.1f KB %.4f | int8 %.1f KB %.4f | size %.2fx | latency %.2fx"
          % (size_fp32 / 1024, v_fp32, size_int8 / 1024, v_int8,
             size_int8 / size_fp32, lat_int8 / lat_fp32), flush=True)
    return res


def run_selfdistill(args, out_dir):
    tok, train_ids, val_ids, data_meta = make_data(args.block_size)
    print("[data] %s" % data_meta, flush=True)
    bs, block = args.batch_size, args.block_size
    # 同结构教师：C=64，训练 300 步（默认学生只有 120 步）
    teacher_cfg = dict(args.student_cfg)
    teacher = new_model(tok.vocab_size, teacher_cfg, args.seed)
    print("[self-teacher] params = %.3f M" % (teacher.n_param / 1e6), flush=True)
    th = train(teacher, train_ids, args.self_teacher_iters, bs, block,
               lr=args.lr, seed=args.seed, log_every=args.log_every,
               tag="self-teacher", record_parts=False)
    tv = evaluate(teacher, val_ids, bs, block)
    torch.save(teacher.state_dict(), out_dir / "teacher_self.pt")
    trow = {"params": teacher.n_param, "val_loss": tv, "ppl": math.exp(tv),
            "time_s": th["time_s"], "curve": th["loss"], "cfg": teacher_cfg,
            "iters": args.self_teacher_iters}

    s0 = new_model(tok.vocab_size, args.student_cfg, args.seed + 7)
    h0 = train(s0, train_ids, args.student_iters, bs, block, lr=args.lr,
               seed=args.seed, log_every=args.log_every, tag="self-scratch",
               record_parts=True)
    row0 = eval_student(s0, h0, "scratch", teacher, val_ids, bs, block,
                        {"T": None, "alpha": 0.0})

    s1 = new_model(tok.vocab_size, args.student_cfg, args.seed + 7)
    h1 = train(s1, train_ids, args.student_iters, bs, block, lr=args.lr,
               seed=args.seed, teacher=teacher, T=args.T, alpha=args.alpha,
               log_every=args.log_every, tag="self-kd", record_parts=True)
    row1 = eval_student(s1, h1, "self_kd", teacher, val_ids, bs, block,
                        {"T": args.T, "alpha": args.alpha})
    torch.save(s1.state_dict(), out_dir / "student_self_kd.pt")
    res = {"data": data_meta, "teacher": trow, "scratch": row0,
           "self_kd": row1,
           "relative_drop_pct": 100 * (row0["val_loss"] - row1["val_loss"]) / row0["val_loss"],
           "env": env_snapshot()}
    save_json(res, out_dir / "results_self.json")
    return res


def run_scan(args, out_dir):
    tok, train_ids, val_ids, data_meta = make_data(args.block_size)
    print("[data] %s" % data_meta, flush=True)
    rows = []
    for C in [96, 160, 256]:
        sub = out_dir / ("C%d" % C)
        sub.mkdir(parents=True, exist_ok=True)
        cfg = make_cfg(args.teacher_layers, args.n_head, C, args.block_size)
        print("\n[scan] teacher C=%d" % C, flush=True)
        teacher, trow = load_or_train_teacher(
            args, tok, train_ids, val_ids, args.batch_size, args.block_size,
            sub, cfg=cfg, allow_formal_fallback=False)
        bs, block = args.batch_size, args.block_size
        s0 = new_model(tok.vocab_size, args.student_cfg, args.seed + 7)
        h0 = train(s0, train_ids, args.student_iters, bs, block, lr=args.lr,
                   seed=args.seed, log_every=args.log_every, tag="scratch-C%d" % C,
                   record_parts=False)
        row0 = eval_student(s0, h0, "scratch-C%d" % C, teacher, val_ids, bs, block,
                            {"T": None, "alpha": 0.0})
        s1 = new_model(tok.vocab_size, args.student_cfg, args.seed + 7)
        h1 = train(s1, train_ids, args.student_iters, bs, block, lr=args.lr,
                   seed=args.seed, teacher=teacher, T=args.T, alpha=args.alpha,
                   log_every=args.log_every, tag="kd-C%d" % C, record_parts=False)
        row1 = eval_student(s1, h1, "kd-C%d" % C, teacher, val_ids, bs, block,
                            {"T": args.T, "alpha": args.alpha})
        rows.append({"teacher_embd": C, "teacher": trow,
                     "scratch": row0, "kd": row1,
                     "relative_drop_pct": 100 * (row0["val_loss"] - row1["val_loss"]) / row0["val_loss"]})
        save_json({"data": data_meta, "row": rows[-1]},
                  sub / ("results_scan_C%d.json" % C))
    res = {"data": data_meta, "rows": rows, "env": env_snapshot()}
    save_json(res, out_dir / "results_scan.json")
    print("[scan] " + " | ".join("C=%d gain=%.2f%%" %
                                 (r["teacher_embd"], r["relative_drop_pct"]) for r in rows),
          flush=True)
    return res


# ============================================================
# 8. 入口
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["quick", "teacher", "all", "ablate", "feat",
                             "quant", "selfdistill", "scan"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--teacher-iters", type=int, default=200)
    ap.add_argument("--student-iters", type=int, default=120)
    ap.add_argument("--abl-iters", type=int, default=80)
    ap.add_argument("--self-teacher-iters", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--block-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--feat", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--teacher-layers", type=int, default=2)
    ap.add_argument("--student-layers", type=int, default=2)
    ap.add_argument("--teacher-embd", type=int, default=160)
    ap.add_argument("--student-embd", type=int, default=64)
    ap.add_argument("--teacher-path", default=None,
                    help="复用指定 teacher.pt；默认 out_dir/teacher.pt")
    ap.add_argument("--student-path", default=None,
                    help="量化模式载入的蒸馏学生权重；默认 results/formal/student_kd.pt")
    ap.add_argument("--retrain", action="store_true",
                    help="忽略已有 teacher.pt，重新训练教师")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    if args.mode == "quick":
        args.teacher_iters = 30
        args.student_iters = 20
        args.abl_iters = 10
        args.batch_size = 8
        args.log_every = 10
    args.teacher_cfg = make_cfg(args.teacher_layers, args.n_head,
                                args.teacher_embd, args.block_size)
    args.student_cfg = make_cfg(args.student_layers, args.n_head,
                                args.student_embd, args.block_size)
    args.out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_DIRS[args.mode]
    return args


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.mode, out_dir, args)
    save_json(env_snapshot(), out_dir / "env.json")
    print("[env] %s" % env_snapshot(), flush=True)
    try:
        if args.mode == "quick":
            res = run_all(args, out_dir)
        elif args.mode == "teacher":
            res = run_teacher(args, out_dir)
        elif args.mode == "all":
            res = run_all(args, out_dir)
        elif args.mode == "ablate":
            res = run_ablate(args, out_dir)
        elif args.mode == "feat":
            res = run_feat(args, out_dir)
        elif args.mode == "quant":
            res = run_quant(args, out_dir)
        elif args.mode == "selfdistill":
            res = run_selfdistill(args, out_dir)
        elif args.mode == "scan":
            res = run_scan(args, out_dir)
        else:
            raise ValueError(args.mode)
        logger.finish("success")
        print("\n[done] 结果已写入 %s" % out_dir, flush=True)
        return res
    except Exception:
        err = traceback.format_exc()
        print(err, flush=True)
        logger.finish("failed", err)
        raise


if __name__ == "__main__":
    main()
