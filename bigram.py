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
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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
blocksize = 8
x = train_data[:blocksize]
y = train_data[1:blocksize+1] # offset by one character because y are the targets for each position
for t in range(blocksize):
    context = x[:t+1]
    target = y[t]
    # print(f"input: {context}, output: {target}")
    
batchsize = 32

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

xb, yb = get_batch('train')

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

class BigramLanguageModel(nn.Module):
    '''This is a simple language model that predicts the next token given the current token.
    The tokens are not 'talking' to each other'''
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # This is a PyTorch Embedding layer (nn.Embedding), which is essentially a lookup table 
        # where each token index corresponds to a vector (embedding)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        # defines the forward pass of the neural network
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (Batch, Time, Channel)
        
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
            logits, loss = self(idx)
            # focus only on the last time step, so only the last time step from T
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # get 1 sample from the distribution; for every batch, we get a single prediction
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)
# logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)
# idx = torch.zeros((1, 1), dtype=torch.long)
# print(decode(m.generate(idx = idx, max_new_tokens=100)[0].tolist()))

'''Create a PyTorch optimizer'''
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

'''Train the model to reduce the loss over iterations'''
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
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