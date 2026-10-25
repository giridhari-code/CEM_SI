# CEM architecture — **no Transformer**

CEM is **not** a Transformer LLM.

| Forbidden in core | Used instead |
|-------------------|--------------|
| Multi-head attention / GQA | BiLSTM, GRU, CNN, MLP |
| RoPE + KV attention cache | Recurrent state (h, c) |
| Fixed depth stack only | **Looped** L3–L6 controlled by L7 meta-cognition |

## Stack

```text
L1 multimodal I/O (text / optional audio-image)
L1.5–L2 fusion
L3–L6 looped core (predictive, modularity, world model, active inference)
L7 meta-cognition (stop / continue)
L8 workspace / self-model
```

Optional helpers under `architecture/`: **GRUBlock**, **LSTMBlock**, **RecurrentState** only.
