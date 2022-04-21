# Dynamic Model Tree
This repository contains an implementation of the Dynamic Model Tree (DMT) that is introduced in

*Haug, Johannes, Klaus Broelemann, and Gjergji Kasneci. "Dynamic Model Tree for Interpretable Data Stream Learning." arXiv preprint arXiv:2203.16181 (2022).*

The DMT has been accepted for the research track of the **International Conference on Data Engineering 2022 (ICDE)**. The paper has not yet been officially published.
In the meantime, an archived version of the paper can be found on [ArXiv](https://arxiv.org/abs/2203.16181). Please refer to our paper when using this implementation.

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