# get our models
# get our dataset
from simplellm.dataloaders import TinyStories
from simplellm.llama import CausalLLama, LLama
# our loss
from simplellm.losses import causalLLMLoss
# get our tokenizer
from simplellm.tokenizers import SPTokenizer
from torch.optim import AdamW

dmodel = 288
num_heads = 6
n_layers = 6
seq_l = 256
batch_size = 3
device = "cuda"

# make the tokenizer
tokenizer = SPTokenizer()
# make the model
net = LLama(
    CausalLLama, tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
    device=device, n_layers=n_layers, ctx_size=seq_l, padding_idx=tokenizer.pad_id)
ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)
# we can iterate the dataset with:
iter_ds = iter(ds)
optim = AdamW(net.parameters(), lr=8e-4)

for itr in range(5_000):
    optim.zero_grad()
    x = next(iter_ds)
    x = x.to(device)
    target = x.clone().detach()
    x = net(x)
    loss = causalLLMLoss(x, target, tokenizer.vocab_size)
    # log the loss
    print(itr, loss.item())
    loss.backward()
    optim.step()
