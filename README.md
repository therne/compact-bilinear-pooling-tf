# Compact Bilinear Pooling for TensorFlow
TensorFlow Implementation of [Compact Bilinear Pooling](https://arxiv.org/abs/1511.06062).

### Prerequisites

- Linux. OS X and Windows are not supported yet :(
- GCC 4.x or higher

### Installation

First, you need to build native op.
```
./build.sh
```

Then, you can use CBP.
```python
import tensorflow as tf
from count_sketch import bilinear_pool

with tf.Session() as sess:
  p1 = tf.constant([[0.83, 0.50, 0.10, 0.82, 0.24, 0.11, 0.21, 0.51, 0.39, 0.10, 0.67, 0.18]])
  p2 = tf.constant([[0.18, 0.59, 0.56, 0.12, 0.33, 0.93, 0.17, 0.58, 0.16]])
  project_size = 6
  
  pooled = bilinear_pool(p1, p2, project_size)

  sess.run(tf.initialize_all_variables())
  print('Result: {}'.format(sess.run(pooled)))
```
