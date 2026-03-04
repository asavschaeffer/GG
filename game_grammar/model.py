"""Value autograd + GPT, adapted from microgpt.py for game-grammar tokens."""

import math
import random

# ── Autograd ──────────────────────────────────────────────────────────────────

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# ── Model functions ───────────────────────────────────────────────────────────

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim, n_embd):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits


# ── Model init / train / sample API ──────────────────────────────────────────

class GameGPT:
    def __init__(self, vocab_size=74, n_layer=2, n_embd=32, block_size=64, n_head=4, seed=42):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.seed = seed

        rng = random.Random(seed)
        matrix = lambda nout, nin, std=0.08: [
            [Value(rng.gauss(0, std)) for _ in range(nin)] for _ in range(nout)
        ]
        self.state_dict = {
            'wte': matrix(vocab_size, n_embd),
            'wpe': matrix(block_size, n_embd),
            'lm_head': matrix(vocab_size, n_embd),
        }
        for i in range(n_layer):
            self.state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
            self.state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
            self.state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
            self.state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
            self.state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
            self.state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

        self.params = [p for mat in self.state_dict.values() for row in mat for p in row]

        # Adam buffers
        self.m = [0.0] * len(self.params)
        self.v = [0.0] * len(self.params)
        self.step_count = 0

    def forward(self, token_id, pos_id, keys, values):
        return gpt(
            token_id, pos_id, keys, values,
            self.state_dict, self.n_layer, self.n_head, self.head_dim, self.n_embd,
        )

    def fresh_kv(self):
        return [[] for _ in range(self.n_layer)], [[] for _ in range(self.n_layer)]

    def train_step(self, tokens, lr=0.01, beta1=0.85, beta2=0.99, eps=1e-8):
        n = min(self.block_size, len(tokens) - 1)
        if n <= 0:
            return 0.0

        keys, values = self.fresh_kv()
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = self.forward(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)

        loss = (1 / n) * sum(losses)
        loss.backward()

        self.step_count += 1
        lr_t = lr * (1 - self.step_count / max(self.step_count + 1, 5000))
        for i, p in enumerate(self.params):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * p.grad
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * p.grad ** 2
            m_hat = self.m[i] / (1 - beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - beta2 ** self.step_count)
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)
            p.grad = 0

        return loss.data

    def sample(self, bos_id, eos_id, temperature=0.5, max_len=None):
        if max_len is None:
            max_len = self.block_size
        keys, values = self.fresh_kv()
        token_id = bos_id
        result = [bos_id]
        for pos_id in range(max_len - 1):
            logits = self.forward(token_id, pos_id, keys, values)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(self.vocab_size), weights=[p.data for p in probs])[0]
            result.append(token_id)
            if token_id == eos_id:
                break
        return result

    def save_weights(self, path):
        """Save model weights as plain text."""
        with open(path, 'w') as f:
            for name, mat in self.state_dict.items():
                for r, row in enumerate(mat):
                    vals = ' '.join(f'{p.data:.8f}' for p in row)
                    f.write(f'{name}|{r}|{vals}\n')

    def load_weights(self, path):
        """Load model weights from plain text."""
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                name, r, vals = line.split('|', 2)
                r = int(r)
                values = [float(v) for v in vals.split()]
                for j, v in enumerate(values):
                    self.state_dict[name][r][j].data = v
