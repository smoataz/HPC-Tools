# Effect of Parallelization on Machine Learning / Deep Learning Training Speed

This project investigates the effect of parallelization on the training speed of machine learning and deep learning models, using the MNIST dataset and a simple convolutional neural network as a case study. We explore the efficiency gains across different hardware configurations using TensorBoard for detailed statistics.

## Hardware Configurations and Performance:

### 1 Node, 1 GPU:

- Nodes: 1
- Cores per node: 32
- CPU Utilized: 00:07:43
- CPU Efficiency: 7.27%
- Job Wall-clock time: 00:03:19
- Memory Utilized: 3.03 GB
- Memory Efficiency: 37.86%

### 1 Node, 2 GPUs:

- Nodes: 1
- Cores per node: 64
- CPU Utilized: 00:10:12
- CPU Efficiency: 9.02%
- Job Wall-clock time: 00:01:46
- Memory Utilized: 5.88 GB (estimated maximum)
- Memory Efficiency: 73.45%

### 2 Nodes, 2 GPUs:

- Nodes: 2
- Cores per node: 64
- CPU Utilized: 00:16:07
- CPU Efficiency: 10.64%
- Job Wall-clock time: 00:01:11
- Memory Utilized: 11.90 GB (estimated maximum)
- Memory Efficiency: 74.38%

## Observations:
- All 3 configuartion Converged the model around 98 % acc
- Increasing the number of GPUs significantly reduces training time:
  - 1 GPU: 3 minutes 19 seconds
  - 2 GPUs (single node): 1 minute 46 seconds (48% faster)
  - 2 GPUs (two nodes): 1 minute 11 seconds (68% faster than 1 GPU)
- CPU utilization and efficiency generally increase with parallelization, indicating better resource utilization.
- Memory usage scales proportionately with GPU count but remains within efficient limits.

## TensorBoard for Detailed Statistics:

We extensively utilize TensorBoard to visualize and analyze training progress and resource utilization across different configuration Take a look at the tb_logs folder 

## Conclusion:

This project demonstrates that parallelization using GPUs effectively accelerates machine learning and deep learning training. Utilizing multiple GPUs on a single node or across multiple nodes offers significant speed improvements while maintaining efficient resource utilization. TensorBoard plays a crucial role in visualizing and analyzing these performance gains, enabling informed decisions about hardware configurations for optimal training efficiency.
