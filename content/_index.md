+++
title = "PTCV21: PyTorch-Ignite"
outputs = ["Reveal"]
[logo]
src = "https://raw.githubusercontent.com/pytorch/ignite/master/assets/logo/ignite_logomark.svg"
+++


# PyTorch-Ignite
> Training and evaluating neural networks
> flexibly and transparently

Victor, Sylvain & Taras

{{< social >}}

---

# Content

- PyTorch-Ignite: what and why?
- Quick-start example
- About the project


---

<!-- Start vertical slides -->
{{% section %}}

# PyTorch-Ignite: what and why?

> High-level **library** to help with training and evaluating neural networks in PyTorch flexibly and transparently.

## Features

- Simple Engine and Event System
- Out-of-the-box metrics to easily evaluate models
- Built-in handlers to compose training pipeline
- Distributed Training support

---

# What makes Pytorch-Ignite unique ?

- Library at the crossroads of 
  - High-level Plug & Play features
  - Under-the-hood expansion possibilities

- Help to promote best pratices 
  - Using simple and coherent concepts
  - Transparently, without magic 
  
- Keep it simple and understandable
---

# Key concepts in a nutshell

- Engine
  - Loop over the user data
  - Apply an arbitrary user function on each data

- Event System
  - Customizable collections of Events
  - Emitting system of Handlers attached to Events

> Reactive programming concepts

---

# In its simplest form

```python
fire_event(Events.STARTED)
while epoch < max_epochs:
    fire_event(Events.EPOCH_STARTED)

    # run once on data
    for batch in data:
        fire_event(Events.ITERATION_STARTED)

        output = process_function(batch)

        fire_event(Events.ITERATION_COMPLETED)
        
    fire_event(Events.EPOCH_COMPLETED)
fire_event(Events.COMPLETED)
```

---

# Do what you want !

```python
def process_function(engine, batch):
    # Do what you want ...
    # ... and return what you want

engine = Engine(process_function)

engine.run(data, max_epochs=50)
```

- A minimalist API for maximum freedom
  - Each Engine contains a sharable State to cache results and outputs

---

# When you want !

```python
@engine.on(Events.STARTED)
def handler():
    # Do what you want when you want !
    # here at every Events.STARTED event triggered by the engine's Event System
```

- Highly flexible and customizable 
  - Custom user Events
  - Event filtering 
  - Handlers can be lambda, objects, functions, methods

---

# Built-in Metrics

- Dedicated to many Deep Learning tasks
- Easily composable to assemble a custom metric 
- Easily extendable to create custom metrics

---

# Built-in Handlers

- Logging to experiment tracking systems
- Optimizer’s parameter scheduling
- Smart checkpointing
- and a lot more !

---

# Distributed training made easy

- Run the same code across all supported backends seamlessly
  - backends from native torch distributed configuration: `nccl`, `gloo`, `mpi`
  - Horovod framework with `gloo` or `nccl` communication backend
  - XLA on TPUs via `pytorch/xla`

```python
import ignite.distributed as idist
```

---

# Distributed launchers

- Handle distributed launchers with the same code
  - `torch.multiprocessing.spawn`
  - `torch.distributed.launch`
  - `horovodrun`
  - `slurm`

```python
def run(_, config):
  # distributed function
  # Do what you want

with idist.Parallel(backend=backend) as parallel:
    parallel.run(run, config)
```

---

# Unified Distributed API

- High-level helper methods
  - `idist.auto_model()` 
  - `idist.auto_optim()` 
  - `idist.auto_dataloader()`
- Collective operations
  - `all_reduce`, `all_gather`, and more

<!-- End vertical slides -->
{{% /section %}}

---

<!-- Start vertical slides -->
{{% section %}}

# Quick-start example

...

---

...

<!-- End vertical slides -->
{{% /section %}}

---

<!-- Start vertical slides -->
{{% section %}}

# About "PyTorch-Ignite" project

Community-driven open source and _NumFOCUS Affiliated_ Project

maintained by volunteers in the PyTorch community:

```
@vfdev-5, @ydcjeff, @KickItLikeShika, @sdesrozis, @alykhantejani, @anmolsjoshi,
@trsvchn, @Moh-Yakoub, ..., @fco-dv, @gucifer, @Priyansi, ...
```

![o1](https://a.slack-edge.com/production-standard-emoji-assets/13.0/apple-medium/1f389@2x.png)
![o2](https://a.slack-edge.com/production-standard-emoji-assets/13.0/apple-medium/1f44f@2x.png)
![o3](https://a.slack-edge.com/production-standard-emoji-assets/13.0/apple-medium/1f64f@2x.png)

With the support of:

<img width="200" src="https://numfocus.org/wp-content/uploads/2017/07/NumFocus_LRG.png">
<img width="200" src="https://raw.githubusercontent.com/Quansight-Labs/quansight-labs-site/master/files/images/QuansightLabs_logo_V2.png">
<img width="200" src="https://pytorch-ignite.ai/images/logos/ifpen.jpg">

---

# Projects using PyTorch-Ignite

- Research Papers
- Blog articles, tutorials, books
- Toolkits
  - [Project MONAI](https://monai.io/), [Nussl](https://nussl.github.io/docs/), ...

More details here: https://pytorch-ignite.ai/ecosystem/

---

# Our Community

- Google Summer of Code 2021
  - Mentored two great students (_Ahmed_ and _Arpan_)

- Google Season of Docs 2021
  - Working with great tech writer (_Priyansi_)

- Hacktoberfest 2020 and coming up 2021

- PyData Global Mentored Sprint 2020

_Stay tuned for upcoming events ..._

<img width="50" src="https://summerofcode.withgoogle.com/static/img/summer-of-code-logo.svg">
<img width="50" src="https://developers.google.com/season-of-docs/images/SeasonofDocs_Icon_Grey_300ppi_trimmed_480.png">
<img width="50" src="https://hacktoberfestswaglist.com/img/Hacktoberfest_20.jpg">
<img width="150" src="https://pydata.org/global2021/wp-content/uploads/2021/06/logo.png">


---

# Join the PyTorch-Ignite Community


We are looking for motivated contributors to help out with the project.

![o1](https://a.slack-edge.com/production-standard-emoji-assets/13.0/apple-small/1f3c5@2x.png)
Everyone is welcome to contribute
![o2](https://a.slack-edge.com/production-standard-emoji-assets/13.0/apple-small/1f4af@2x.png)

<a class="level-item" href="https://www.github.com/pytorch/ignite">
    <span class="icon"><i class="fab fa-github"></i></span>
</a>

<a class="level-item" href="https://discord.gg/djZtm3EmKj">
    <span class="icon"><i class="fab fa-discord"></i></span>
</a>

How to start:

- Read our [Contributing guides](https://github.com/pytorch/ignite/blob/master/CONTRIBUTING.md)
- Pick [List of Help-wanted GH issue](https://github.com/pytorch/ignite/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
- Reach out to us on GH or Discord for more guidance

---

Follow us on

<a class="level-item" href="https://www.twitter.com/pytorch_ignite">
    <span class="icon"><i class="fab fa-twitter"></i></span>
</a>

<a class="level-item" href="https://www.facebook.com/PyTorch-Ignite-Community-105837321694508">
    <span class="icon"><i class="fab fa-facebook"></i></span>
</a>

<a class="level-item" href="https://dev.to/pytorch-ignite">
    <span class="icon"><i class="fab fa-dev"></i></span>
</a>

and check out our new website:

https://pytorch-ignite.ai

<!-- End vertical slides -->
{{% /section %}}
