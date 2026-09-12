# cpugrad

cpugrad is a c++ machine learning library focused on high performance training on CPUs.

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

The goal is for this library to be general purpose. At this early stage, though, there are likely many things that are hard-coded to fit my specific needs when they should be configurable.

Some simple test networks are provided in the [cpugrad-test](https://github.com/mikeoliphant/cpugrad/tree/main/cpugrad-test) folder. For a more complex example, see my [CPU trainer for neural amp modeler](https://github.com/mikeoliphant/nam-cpu-trainer).
