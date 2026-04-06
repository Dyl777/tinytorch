import torch
from torch.nn import Linear, Embedding
import torch.nn.functional as F
from torch.optim.sgd import SGD
import numpy as np


def load_tensor(fname: str, *shape: int, dtype=np.float32):
    return torch.tensor(np.fromfile(fname, dtype=dtype)).view(*shape)


def main():
    batch_size = 32
    seq_len = 16
    emb_size = 16
    hidden_size = 4 * emb_size
    vocab_size = 27

    Xs = load_tensor("dump/Xs", batch_size * seq_len, dtype=np.int32).to(torch.int64)
    Ys = load_tensor("dump/Ys", batch_size, dtype=np.int32).to(torch.int64)

    wte = Embedding(vocab_size, emb_size)
    wte.weight.data = load_tensor("dump/wte", vocab_size, emb_size)
    wte_out = wte(Xs)

    fc_1 = Linear(seq_len * emb_size, hidden_size)
    fc_1.weight.data = load_tensor("dump/fc_1.w", seq_len * emb_size, hidden_size).T
    fc_1.bias.data = load_tensor("dump/fc_1.b", hidden_size)
    fc_1_out = fc_1(wte_out.view(batch_size, seq_len * emb_size))
    fc_1_relu = F.relu(fc_1_out)

    fc_2 = Linear(hidden_size, vocab_size)
    fc_2.weight.data = load_tensor("dump/fc_2.w", hidden_size, vocab_size).T
    fc_2.bias.data = load_tensor("dump/fc_2.b", vocab_size)
    fc_2_out = fc_2(fc_1_relu)
    fc_2_softmax = F.softmax(fc_2_out, dim=-1)

    torch.testing.assert_close(wte_out, load_tensor("dump/wte.out", batch_size * seq_len, emb_size))
    torch.testing.assert_close(fc_1_out, load_tensor("dump/fc_1.out", batch_size, hidden_size))
    torch.testing.assert_close(fc_1_relu, load_tensor("dump/fc_1.relu", batch_size, hidden_size))
    torch.testing.assert_close(fc_2_out, load_tensor("dump/fc_2.out", batch_size, vocab_size))
    torch.testing.assert_close(fc_2_softmax, load_tensor("dump/fc_2.softmax", batch_size, vocab_size))

    loss = F.cross_entropy(fc_2_out, Ys)
    print(loss.item())

    loss.backward()
    torch.testing.assert_close(fc_2.weight.grad, load_tensor("dump/fc_2.d_w", hidden_size, vocab_size).T)
    torch.testing.assert_close(fc_2.bias.grad, load_tensor("dump/fc_2.d_b", vocab_size))
    torch.testing.assert_close(fc_1.weight.grad, load_tensor("dump/fc_1.d_w", seq_len * emb_size, hidden_size).T)
    torch.testing.assert_close(fc_1.bias.grad, load_tensor("dump/fc_1.d_b", hidden_size))
    torch.testing.assert_close(wte.weight.grad, load_tensor("dump/d_wte", vocab_size, emb_size))


if __name__ == "__main__":
    main()
