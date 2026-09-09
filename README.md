# cpugrad

cpugrad is a c++ machine learning library focused on high performance on CPUs.

It is currently very much a work in progress. Currently it is mostly targeting time series data (ie: audio samples).

Features so far:

- Self-contained with minimal dependencies
- Dense layers
- Causal 1D convolution layers
- Threaded mini-batching
- MSE and ESR loss
- "Adam" optimizer
- Arena-based memory buffer allocation

It is currently very much a not-so-autograd. Backward passes must be hand-written. This may change in the future.
