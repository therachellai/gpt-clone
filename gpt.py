import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337) # ensure it is the same random number every time

'''Download data with the following command: wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'''
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
# print("length of dataset in characters: ", len(text))
# print(text[:1000])

'''Unique characters that occur in the text'''
chars = sorted(list(set(text)))
vocab_size = len(chars)
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 6
n_layer = 6
dropout = 0.2
# print(''.join(chars))
# print(vocab_size)

'''Translating characters into integers using simple encoding and decoding functions (character-level tokenizers)'''
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

'''Encode the text dataset and store it in torch.Tensor'''
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

'''Split the training and testing data'''
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

'''Train one chunk of the entire dataset at a time'''
blocksize = 256
x = train_data[:blocksize]
y = train_data[1:blocksize+1] # offset by one character because y are the targets for each position
for t in range(blocksize):
    context = x[:t+1]
    target = y[t]
    # print(f"input: {context}, output: {target}")
    
batchsize = 64

def get_batch(split):
    """Generate a small batch of data of input x and target y

    Args:
        split (str): _description_
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - blocksize, (batchsize,))
    x = torch.stack([data[i:i+blocksize] for i in ix])
    y = torch.stack([data[i+1:i+blocksize+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # tells pytorch that we don't need to do backpropagation
def estimate_loss():
    '''Average loss over a few batches of data'''
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(blocksize, blocksize)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    '''This is a simple language model that predicts the next token given the current token.
    The tokens are not 'talking' to each other'''
    
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # This is a PyTorch Embedding layer (nn.Embedding), which is essentially a lookup table 
        # where each token index corresponds to a vector (embedding)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(blocksize, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # defines the forward pass of the neural network
        # idx and targets are both (B,T) tensor of integers
        tok_embed = self.token_embedding_table(idx) # (Batch, Time, Channel)
        pos_embed = self.positional_embedding_table(torch.arange(T, device=device)) # (Time, Channel)
        x = tok_embed + pos_embed # (Batch, Time, Channel)
        x = self.blocks(x) # (Batch, Time, Channel)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (Batch, Time, Channel=vocab_size) This C is not the same as the previous C
        
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # stretch out the array so that it becomes 2D
            targets = targets.view(B*T) # do the same things for targets
            loss = F.cross_entropy(logits, targets) # computes the cross-entropy loss between the predicted logits and the target labels
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the latest blocksize tokens
            idx_cond = idx[:, -blocksize:]
            logits, loss = self(idx_cond)
            # focus only on the last time step, so only the last time step from T
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # get 1 sample from the distribution; for every batch, we get a single prediction
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = GPTLanguageModel()
m = model.to(device)
# logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)
# idx = torch.zeros((1, 1), dtype=torch.long)
# print(decode(m.generate(idx = idx, max_new_tokens=100)[0].tolist()))

'''Create a PyTorch optimizer'''
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

'''Train the model to reduce the loss over iterations'''
fm = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))