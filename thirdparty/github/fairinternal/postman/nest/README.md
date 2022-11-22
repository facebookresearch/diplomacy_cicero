
# Nest library

```shell
CXX=c++ pip install . -vv
```

Usage in Python:

```python
import torch
import nest

t1 = torch.tensor(0)
t2 = torch.tensor(1)
d = {'hey': torch.tensor(2)}

print(nest.map(lambda t: t + 42, (t1, t2, d)))
# --> (tensor(42), tensor(43), {'hey': tensor(44)})
```
