import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

book = open('moby_dick.txt','r',encoding='utf-8')
book = book.read()

chars = sorted(list(set(book)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Using character level tokenizer instead subword encoding tokenizer because low number of chars
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take a list of integers, output a string

# print(encode("Random String"))
# print(decode(encode("Random String")))

# Encode entire text dataset and store inside a torch.Tensor
data = torch.tensor(encode(book),dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

n = int(0.9+len(data))
train_data = data[:n]
val_data = data[n:]

# block_size = 8
# train_data[:block_size+1]

# # first 90% of data will be trained, rest will be validation sets
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target: {target}")

torch.manual_seed(1337)

batch_size = 4 # how many independent sequences will process in parallel
block_size = 8 # maximum context length for predictions

def get_batch(split):
    # generate small batch of data of inputs x and y targets
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")

class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
    
    def forward(self,idx,targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) batch, time, channel
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets) # How well will predict next target according to logits
            
        return logits, loss
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            #focus only on the last time step
            logits = logits[:,-1,:] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits,dim=-1) #(B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs,num_samples=1) # (B,1)
            #append sample index to the running sequence
            idx = torch.cat((idx,idx_next),dim=1) # (B,T+1)
        return idx
    

m = BigramLanguageModel(vocab_size)
# out = m(xb,yb)
logits, loss = m(xb,yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1,1),dtype=torch.long),max_new_tokens=100)[0].tolist()))
