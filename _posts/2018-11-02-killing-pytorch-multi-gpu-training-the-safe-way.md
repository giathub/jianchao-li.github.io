---
layout: post
comments: true
title:  "Killing PyTorch Multi-GPU Training the Safe Way"
excerpt: "Recently I was working with PyTorch multi-GPU training and I came across a nightmare GPU memory problem. After some expensive trial and error, I finally found a solution for it."
date:   2018-11-02 14:58:10 +0800
mathjax: true
---

As you may have noticed from the title, this post is somewhat different from my previous ones. I would like to talk about a PyTorch DataLoader issue I encountered recently. I feel like devoting a post to it because it has taken me long time to figure out how to fix it.

## Memory consumption

Since my last post on [FCNs](https://jianchao-li.github.io/2018/09/15/understanding-fully-convolutional-networks/), I have been working on semantic segmentation. Nowadays, we have deep neural networks for it, like the state-of-the-art [PSPNet](https://arxiv.org/abs/1612.01105) from CVPR 2017.

In practice, segmentation networks are **much more memory-intensive** than recognition/classification networks. The reason is that semantic segmentation requires dense pixel-level predictions. For example, in the ImageNet classification task, you may use a neural network to transform a 224x224 image into 1000 real numbers (class probabilities). However, in semantic segmentation, suppose you have 20 semantic classes, you need to transform the 224x224 image into 20 224x224 probability maps, each representing probabilities of pixels belonging to one class. The output size changes from 1000 to 20x224x224=1003520, which is more than 1000 times!

Besides the output, the intermediate feature maps in segmentation networks also consume more memory. In recognition networks, sizes of intermediate feature maps usually decrease monotically. However, since segmentation requires output of the same spatial dimension as the input, the feature maps will go through an extra process with their sizes increased (upsampled) back to the size of the input image. This extra upsample process further increases the memory consumption of segmentation networks.

So, when we fit segmentation networks on a GPU, we need to reduce the batch size of the data. However, batch size is crucial to the performance of networks, especially those containing the batch normalization layer. Since no more data can be held in a single GPU, a natural soltuion is to use multiple GPUs and split the data across them (or more formally, *data parallelism*).

## Synchronized batch normalization

Here I would like to make a digression and mention an interesting layer, the synchronized batch normalization layer, which is introduced to increase the *working batch size* for multi-GPU training. You may refer to the section **Cross-GPU Batch Normalization** in [MegDet](https://arxiv.org/abs/1711.07240) for more details.

When we use data parallelism to train on multiple GPUs, a batch of images will be splitted across several GPUs. Suppose your batch size is 16 (a common setting in semantic segmentation) and you train on 8 GPUs with data parallelism, then each GPU will have 2 images. A normal batch norm layer will only uses the 2 images on a single GPU to compute the mean and standard deviation, which is highly inaccurate and will make the training unstable.

To effectively increase the working batch size, we need to synchronize all the GPUs in the batch norm layer, and fetch the mean and standard deviation computed at each GPU to compute a global value using all images. This is what synchronized batch norm layer does. If you would like to learn more about its implementation details, you may have a look at [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch).

## The Issue

After so much background information, the main idea is that semantic segmentation networks are very memory-intensive and require multiple GPUs to train a reasonable batch size. And synchronized batch norm can be used to increase the working batch size in multi-GPU training.

Now comes the issue that I encountered recently. I was working with a semantic segmentation codebase written in PyTorch on a machine with 8 GPUs. The codebase incorporates synchronized batch norm and uses PyTorch multiprocessing for its custom DataLoader. I ran the training program for some time and then I killed it (I was running the program in a virtualized docker container in a cloud GPU cluster. So killing it is just to click a button in the cloud GUI).

Then I checked the GPUs using `nvidia-smi` and everything looked good.

```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.46                 Driver Version: 390.46                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:1A:00.0 Off |                    0 |
| N/A   32C    P0    25W / 250W |     11MiB / 16160MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE...  Off  | 00000000:1F:00.0 Off |                    0 |
| N/A   34C    P0    25W / 250W |     11MiB / 16160MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-PCIE...  Off  | 00000000:20:00.0 Off |                    0 |
| N/A   33C    P0    25W / 250W |     11MiB / 16160MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-PCIE...  Off  | 00000000:21:00.0 Off |                    0 |
| N/A   33C    P0    23W / 250W |     11MiB / 16160MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   4  Tesla V100-PCIE...  Off  | 00000000:B2:00.0 Off |                    0 |
| N/A   32C    P0    26W / 250W |     11MiB / 16160MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   5  Tesla V100-PCIE...  Off  | 00000000:B3:00.0 Off |                    0 |
| N/A   35C    P0    26W / 250W |     11MiB / 16160MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   6  Tesla V100-PCIE...  Off  | 00000000:B4:00.0 Off |                    0 |
| N/A   34C    P0    25W / 250W |     11MiB / 16160MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   7  Tesla V100-PCIE...  Off  | 00000000:B5:00.0 Off |                    0 |
| N/A   35C    P0    25W / 250W |     11MiB / 16160MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

But then when I tried to start a new training program. An OOM error occurred. For the sake of privacy, some traceback logs were omitted by `...`.

```bash
Traceback (most recent call last):
  ...
  File "/usr/local/lib/python3.6/site-packages/torch/cuda/streams.py", line 21, in __new__
    return super(Stream, cls).__new__(cls, priority=priority, **kwargs)
RuntimeError: CUDA error (2): out of memory
Exception in thread Thread-1:
Traceback (most recent call last):
  File "/usr/local/lib/python3.6/threading.py", line 916, in _bootstrap_inner
    self.run()
  File "/usr/local/lib/python3.6/threading.py", line 864, in run
    self._target(*self._args, **self._kwargs)
  ...
  File "/usr/local/lib/python3.6/multiprocessing/queues.py", line 337, in get
    return _ForkingPickler.loads(res)
  File "/usr/local/lib/python3.6/site-packages/torch/multiprocessing/reductions.py", line 151, in rebuild_storage_fd
    fd = df.detach()
  File "/usr/local/lib/python3.6/multiprocessing/resource_sharer.py", line 57, in detach
    with _resource_sharer.get_connection(self._id) as conn:
  File "/usr/local/lib/python3.6/multiprocessing/resource_sharer.py", line 87, in get_connection
    c = Client(address, authkey=process.current_process().authkey)
  File "/usr/local/lib/python3.6/multiprocessing/connection.py", line 493, in Client
    answer_challenge(c, authkey)
  File "/usr/local/lib/python3.6/multiprocessing/connection.py", line 732, in answer_challenge
    message = connection.recv_bytes(256)         # reject large message
  File "/usr/local/lib/python3.6/multiprocessing/connection.py", line 216, in recv_bytes
    buf = self._recv_bytes(maxlength)
  File "/usr/local/lib/python3.6/multiprocessing/connection.py", line 407, in _recv_bytes
    buf = self._recv(4)
  File "/usr/local/lib/python3.6/multiprocessing/connection.py", line 379, in _recv
    chunk = read(handle, remaining)
ConnectionResetError: [Errno 104] Connection reset by peer
```

I ran `nvidia-smi` again and everything still seemed good. So I wrote a `check.cu` to check the GPU memory using CUDA APIs.

```cpp
#include <iostream>
#include "cuda.h"
#include "cuda_runtime_api.h"
  
using namespace std;
  
int main( void ) {
    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount( &num_gpus );
    for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        cudaSetDevice( gpu_id );
        int id;
        cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        cout << "GPU " << id << " memory: free=" << free << ", total=" << total << endl;
    }
    return 0;
}
```

Again, everything looked good.

```bash
$ nvcc check.cu -o check && ./check
GPU 0 memory: free=16488464384, total=16945512448
GPU 1 memory: free=16488464384, total=16945512448
GPU 2 memory: free=16488464384, total=16945512448
GPU 3 memory: free=16488464384, total=16945512448
GPU 4 memory: free=16488464384, total=16945512448
GPU 5 memory: free=16488464384, total=16945512448
GPU 6 memory: free=16488464384, total=16945512448
GPU 7 memory: free=16488464384, total=16945512448
```

Since the error happened to PyTorch, I moved on to write a `check.py`, which created a single-element PyTorch CUDA tensor for sanity check. And this script reproduced the OOM error.

```python
import torch
 
if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        try:
	    torch.cuda.set_device(gpu_id)
	    torch.randn(1, device='cuda')
	    print('GPU {} is good'.format(gpu_id))
        except Exception as exec:
            print('GPU {} is bad: {}'.format(gpu_id, exec))
```

The output was as follows: GPU 1 and 2 were OOM.

```bash
$ python3 check.py 
GPU 0 is good
GPU 1 is bad: CUDA error: out of memory
THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=663 error=2 : out of memory
GPU 2 is bad: cuda runtime error (2) : out of memory at /pytorch/aten/src/THC/THCGeneral.cpp:663
GPU 3 is good
GPU 4 is good
GPU 5 is good
GPU 6 is good
GPU 7 is good
```

So, my GPUs 2 and 3 should be magically occupied by some zombie process. And I had to restart the machine to fix it. I think the zombie process was generated due to my incorrect way of killing the training program. So I decided not to use the kill button in the cloud GUI but logged into the docker container to kill it in the terminal.

I searched on Google for how to kill a PyTorch multi-GPU training program. And I found [@smth](https://discuss.pytorch.org/u/smth/summary)'s suggestion in [this reply](https://discuss.pytorch.org/t/pytorch-doesnt-free-gpus-memory-of-it-gets-aborted-due-to-out-of-memory-error/13775/14?u=jianchao-li).

>@rabst
so, I remember this issue. When investigating, we found that there’s actually a bug in python multiprocessing that might keep the child process hanging around, as zombie processes.
It is not even visible to `nvidia-smi` .
>The solution is `killall python` , or to `ps -elf | grep python` and find them and `kill -9 [pid]` to them.

It explained why `nvidia-smi` failed to reveal the memory issue. Great! But, the above commands did not work for me...

> Nothing is so fatiguing as the eternal haning on of an uncompleted task.
> <div style="text-align: right;">--- William James</div>

## The Solution

After several days of searching, failing, searching again, failing again etc., I finally found one solution. It is just to find out the processes that occupied the GPUs and kill them. To find out those processes, I ran `fuser -v /dev/nvidia*`, which listed all the processes that were occupying my NVIDIA GPUs. Since I have 8 GPUs, the output of this command is a bit log.

```bash
$ fuser -v /dev/nvidia*
                     USER        PID ACCESS COMMAND
/dev/nvidia0:        root       5284 F...m python3
                     root       5416 F...m python3
                     root       5417 F...m python3
                     root       5418 F...m python3
                     root       5419 F...m python3
                     root       5420 F...m python3
                     root       5421 F...m python3
                     root       5422 F...m python3
                     root       5423 F...m python3
                     root       5424 F...m python3
                     root       5425 F...m python3
                     root       5426 F...m python3
                     root       5427 F...m python3
                     root       5428 F...m python3
                     root       5429 F...m python3
                     root       5430 F...m python3
                     root       5431 F...m python3
/dev/nvidia1:        root       5284 F...m python3
                     root       5416 F...m python3
                     root       5417 F...m python3
                     root       5418 F...m python3
                     root       5419 F...m python3
                     root       5420 F...m python3
                     root       5421 F...m python3
                     root       5422 F...m python3
                     root       5423 F...m python3
                     root       5424 F...m python3
                     root       5425 F...m python3
                     root       5426 F...m python3
                     root       5427 F...m python3
                     root       5428 F...m python3
                     root       5429 F...m python3
                     root       5430 F...m python3
                     root       5431 F...m python3
/dev/nvidia2:        root       5284 F...m python3
                     root       5416 F...m python3
                     root       5417 F...m python3
                     root       5418 F...m python3
                     root       5419 F...m python3
                     root       5420 F...m python3
                     root       5421 F...m python3
                     root       5422 F...m python3
                     root       5423 F...m python3
                     root       5424 F...m python3
                     root       5425 F...m python3
                     root       5426 F...m python3
                     root       5427 F...m python3
                     root       5428 F...m python3
                     root       5429 F...m python3
                     root       5430 F...m python3
                     root       5431 F...m python3
/dev/nvidia3:        root       5284 F...m python3
                     root       5416 F...m python3
                     root       5417 F...m python3
                     root       5418 F...m python3
                     root       5419 F...m python3
                     root       5420 F...m python3
                     root       5421 F...m python3
                     root       5422 F...m python3
                     root       5423 F...m python3
                     root       5424 F...m python3
                     root       5425 F...m python3
                     root       5426 F...m python3
                     root       5427 F...m python3
                     root       5428 F...m python3
                     root       5429 F...m python3
                     root       5430 F...m python3
                     root       5431 F...m python3
/dev/nvidia4:        root       5284 F...m python3
                     root       5416 F...m python3
                     root       5417 F...m python3
                     root       5418 F...m python3
                     root       5419 F...m python3
                     root       5420 F...m python3
                     root       5421 F...m python3
                     root       5422 F...m python3
                     root       5423 F...m python3
                     root       5424 F...m python3
                     root       5425 F...m python3
                     root       5426 F...m python3
                     root       5427 F...m python3
                     root       5428 F...m python3
                     root       5429 F...m python3
                     root       5430 F...m python3
                     root       5431 F...m python3
/dev/nvidia5:        root       5284 F...m python3
                     root       5416 F...m python3
                     root       5417 F...m python3
                     root       5418 F...m python3
                     root       5419 F...m python3
                     root       5420 F...m python3
                     root       5421 F...m python3
                     root       5422 F...m python3
                     root       5423 F...m python3
                     root       5424 F...m python3
                     root       5425 F...m python3
                     root       5426 F...m python3
                     root       5427 F...m python3
                     root       5428 F...m python3
                     root       5429 F...m python3
                     root       5430 F...m python3
                     root       5431 F...m python3
/dev/nvidia6:        root       5284 F...m python3
                     root       5416 F...m python3
                     root       5417 F...m python3
                     root       5418 F...m python3
                     root       5419 F...m python3
                     root       5420 F...m python3
                     root       5421 F...m python3
                     root       5422 F...m python3
                     root       5423 F...m python3
                     root       5424 F...m python3
                     root       5425 F...m python3
                     root       5426 F...m python3
                     root       5427 F...m python3
                     root       5428 F...m python3
                     root       5429 F...m python3
                     root       5430 F...m python3
                     root       5431 F...m python3
/dev/nvidia7:        root       5284 F...m python3
                     root       5416 F...m python3
                     root       5417 F...m python3
                     root       5418 F...m python3
                     root       5419 F...m python3
                     root       5420 F...m python3
                     root       5421 F...m python3
                     root       5422 F...m python3
                     root       5423 F...m python3
                     root       5424 F...m python3
                     root       5425 F...m python3
                     root       5426 F...m python3
                     root       5427 F...m python3
                     root       5428 F...m python3
                     root       5429 F...m python3
                     root       5430 F...m python3
                     root       5431 F...m python3
/dev/nvidiactl:      root       5284 F...m python3
                     root       5416 F...m python3
                     root       5417 F...m python3
                     root       5418 F...m python3
                     root       5419 F...m python3
                     root       5420 F...m python3
                     root       5421 F...m python3
                     root       5422 F...m python3
                     root       5423 F...m python3
                     root       5424 F...m python3
                     root       5425 F...m python3
                     root       5426 F...m python3
                     root       5427 F...m python3
                     root       5428 F...m python3
                     root       5429 F...m python3
                     root       5430 F...m python3
                     root       5431 F...m python3
/dev/nvidia-uvm:     root       5284 F.... python3
                     root       5416 F.... python3
                     root       5417 F.... python3
                     root       5418 F.... python3
                     root       5419 F.... python3
                     root       5420 F.... python3
                     root       5421 F.... python3
                     root       5422 F.... python3
                     root       5423 F.... python3
                     root       5424 F.... python3
                     root       5425 F.... python3
                     root       5426 F.... python3
                     root       5427 F.... python3
                     root       5428 F.... python3
                     root       5429 F.... python3
                     root       5430 F.... python3
                     root       5431 F.... python3
```

As can be seen from above, the main process had a PID of 5284. I spawned 16 workers for the DataLoader so there were 16 subprocesses whose PIDs were consecutive (from 5416 to 5431). First I used `kill -9` to kill all of them. Then killed the main process.

```bash
$ for pid in {5416..5431}; do kill -9 $pid; done # kill subprocesses
$ kill -9 5284 # kill main process
```

After killing the subprocesses and main process, I ran `check.py` again and this time every GPU was good.

```bash
$ python3 check.py 
GPU 0 is good
GPU 1 is good
GPU 2 is good
GPU 3 is good
GPU 4 is good
GPU 5 is good
GPU 6 is good
GPU 7 is good
```

## Another Trick

If the above solution still does not work for you (it does happen to me sometimes), the following trick may be helpful. First, find out the training loop of your program. In most cases it will contain a loop based on the number of iterations. Then add the following code to that loop.

```python
if os.path.isfile('kill.me'):
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
    exit(0)
```

Inside the `if` statement, the code empties the caches of all GPUs and exits. After you add this code to the training iteration, once you want to stop it, just `cd` into the directory of the training program and run

```bash
touch kill.me
```

Then in the current or next iteration (based on whether the above code has been executed), the `if` check will become true and all GPUs will be cleared and the program will exit. Since you directly tell Python to exit in the program, it will take care of everything for you. You may use anything instead of `kill.me`. But just make sure it is special enough and thus you will not terminate the training inadvertently by creating a file with the same name.

## Conclusions

The issue made me stuck for long time. And in this process of looking for the solution, I made some expensive trial and error. Several GPU servers in the cloud had a card OOM due to my incorrect way of killing the training program. And I had to ask the administrators to restart them. So I would definitely like others to avoid such a case.

From another perspective, I would like to highlight the importance of engineering capabilities and experiences. Though I was working on semantic segmentation, I spent most of my time digging through all sorts of problems while running the multi-GPU codes.

A final remark, I would not like to leave you an impression that I am blaming the issue on PyTorch, CUDA, the cloud GPU cluster, or any others. Actually it is mainly due to that I do not understand how PyTorch multi-GPU and multiprocessing work. And I think I will need to study these topics more systematically. Hopefully I will write a new post after learning more about them :-)
