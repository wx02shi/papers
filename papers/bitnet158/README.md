# Bitnet 1.58
Whitepaper: [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/pdf/2402.17764.pdf)

## TLDR
Essentially replaces all Linear layers with `BitLinear` layers, which contain quantized weights. More specifically, all parameters are in the range $[-1,0,1]$.

This is a rather extreme strand of quantization, developed with quantization-aware training in mind. The whitepaper reports that a LLaMA-like model with `BitLinear` layers has competitive performance, whilst consuming up to 3.32x less memory and having up to 2.4x lower latency. 

### Constraining Weights
Let $W$ be the standard weights in a linear layer. We obtain a constrained weight matrix by:

$$\widetilde W=\text{RoundClip}\left(\frac{W}{\gamma+\epsilon},-1,1\right)$$
$$\text{RoundClip}(x,a,b)=\max(a,\min(b,\text{round}(x)))$$
$$\gamma=\frac{1}{nm}\sum_{ij}|W_{ij}|$$
where $\epsilon$ is some small floating point number, to prevent overflow when performing this clipping.

### Constraining Activations
With respect to non-linear functions like ReLU, the activations are scaled into the range $[0, Q_b]$, where $Q_b=2^{b-1}$. Typically, $b=8$ (i.e. `int8`).

$$\tilde x=\text{Clip}\left((x-\eta)\times \frac{Q_b}{\gamma},\epsilon, Q_b-\epsilon\right)$$
$$\text{Clip}(x,a,b)=\max(a,\min(b,x))$$
where $\eta=\min_{ij}x_{ij}$ and $\gamma=\|x\|_\infty$.

