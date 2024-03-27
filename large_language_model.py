import torch

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
