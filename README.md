# Dynamic Model Tree
This repository contains an implementation of the Dynamic Model Tree (DMT) that is introduced in

*Johannes Haug, Klaus Broelemann and Gjergji Kasneci, "Dynamic Model Tree for Interpretable Data Stream Learning," 2022 IEEE 38th International Conference on Data Engineering (ICDE), 2022, pp. 2562-2574, doi: 10.1109/ICDE53745.2022.00237.*

The paper can be found on [IEEE](https://ieeexplore.ieee.org/document/9835609) and [arXiv](https://arxiv.org/abs/2203.16181). Please refer to our paper when using this implementation.

## Use the DMT for Online Learning
The DMT implementation provided here uses (multinomial) logit simple models and the negative log-likelihood loss, as described in the paper.
The DMT can be used for both binary and multiclass classification.

The full experiments accompanying the ICDE paper can be found in ``./icde_experiments``.
A simple experiment using the scikit-multiflow framework is given below:

```python
from skmultiflow.data import FileStream
from dmt.DMT import DynamicModelTree
from sklearn.metrics import accuracy_score

# Load data as scikit-multiflow FileStream
stream = FileStream('yourData.csv', target_idx=-1)

# Initial fit (in order to predict observations at the first time step
# of the simulated data stream, the DMT needs to be pre-trained).
model = DynamicModelTree()
x, y = stream.next_sample(batch_size=100)
model.partial_fit(x, y)

while stream.has_more_samples():
    # Note: DMTs work with any batch size.
    x, y = stream.next_sample(batch_size=1)

    y_pred = model.predict(x)
    print(accuracy_score(y, y_pred))
    model.partial_fit(x, y)

stream.restart()
```
