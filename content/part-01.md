+++
weight = 1
+++


<!-- Start vertical slides -->
{{% section %}}

# PyTorch-Ignite: what and why?

> High-level **library** to help with training and evaluating neural networks in PyTorch flexibly and transparently.

- https://github.com/pytorch/ignite


<table style="font-size: 20px;">
<tr>

<td>

```python

def train_step(engine, batch):
  #  ... any training logic ...
  return batch_loss

trainer = Engine(train_step)

# Compose your pipeline ...

trainer.run(train_loader, max_epochs=100)


```
</td>

<td>

```python

metrics = {
  "precision": Precision(),
  "recall": Recall()
}

evaluator = create_supervised_evaluator(
  model,
  metrics=metrics
)


```
</td>

<td>

```python
@trainer.on(Events.EPOCH_COMPLETED)
def run_evaluation():
  evaluator.run(test_loader)

handler = ModelCheckpoint(
  '/tmp/models', 'checkpoint'
)
trainer.add_event_handler(
  Events.EPOCH_COMPLETED,
  handler,
  {'model': model}
)
```
</td>

</tr>

</table>

---

# What makes PyTorch-Ignite unique ?

- Composable and interoperable components
- Simple and understandable code
- Open-source community involvement


<!--
- A library at the crossroads
  - High-level Plug & Play features
  - Expansion possibilities under the hood

- Highly modular
  - Use of simple and consistent concepts
  - Composable and interoperable components
  - Transparent, without magic

- Keep it simple and understandable
  - Contribute to the promotion of best practices
-->

---

# Key concepts in a nutshell

#### PyTorch-Ignite is about:

1) Engine and Event System
2) Out-of-the-box metrics to easily evaluate models
3) Built-in handlers to compose training pipeline
4) Distributed Training support

---

## Engine and Event System

<table style="font-size: 20px;">
<tr>

<td>

{{< add_vspace >}}

- **Engine**
  - Loops on user data
  - Applies an arbitrary user function on batches

- **Event system**
  - Customizable event collections
  - Triggers handlers attached to events

</td>

<td>
In its simpliest form:

```python{2,5,7|1,3,6,8,10,11}
fire_event(Events.STARTED)
while epoch < max_epochs:
    fire_event(Events.EPOCH_STARTED)

    for batch in data:
        fire_event(Events.ITERATION_STARTED)
        output = train_step(batch)
        fire_event(Events.ITERATION_COMPLETED)

    fire_event(Events.EPOCH_COMPLETED)
fire_event(Events.COMPLETED)
```
</td>

</tr>
</table>

---

### Simplified training and validation loop

<div style="font-size: 20px;">

No more coding `for/while` loops on epochs and iterations. Users instantiate engines and run them.

```python
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy


# Setup training engine:
def train_step(engine, batch):
    # Users can do whatever they need on a single iteration
    # Eg. forward/backward pass for any number of models, optimizers, etc
    # ...

trainer = Engine(train_step)

# Setup single model evaluation engine
evaluator = create_supervised_evaluator(model, metrics={"accuracy": Accuracy()})

def validation():
    state = evaluator.run(validation_data_loader)
    # print computed metrics
    print(trainer.state.epoch, state.metrics)

# Run model's validation at the end of each epoch
trainer.add_event_handler(Events.EPOCH_COMPLETED, validation)

# Start the training
trainer.run(training_data_loader, max_epochs=100)
```
</div>

---

### Power of Events & Handlers

#### 1. Execute any number of functions whenever you wish

<div style="font-size: 20px;">

Handlers can be any function: e.g. lambda, simple function, class method, etc.

```python
trainer.add_event_handler(Events.STARTED, lambda _: print("Start training"))

# attach handler with args, kwargs
mydata = [1, 2, 3, 4]
logger = ...

def on_training_ended(data):
    print(f"Training is ended. mydata={data}")
    # User can use variables from another scope
    logger.info("Training is ended")


trainer.add_event_handler(Events.COMPLETED, on_training_ended, mydata)
# call any number of functions on a single event
trainer.add_event_handler(Events.COMPLETED, lambda engine: print(engine.state.times))

@trainer.on(Events.ITERATION_COMPLETED)
def log_something(engine):
    print(engine.state.output)
```
</div>

---

### Power of Events & Handlers

#### 2. Built-in events filtering and stacking

<div style="font-size: 20px;">

```python
# run the validation every 5 epochs
@trainer.on(Events.EPOCH_COMPLETED(every=5))
def run_validation():
    # run validation

@trainer.on(Events.COMPLETED | Events.EPOCH_COMPLETED(every=10))
def run_another_validation():
    # ...

# change some training variable once on 20th epoch
@trainer.on(Events.EPOCH_STARTED(once=20))
def change_training_variable():
    # ...

# Trigger handler with customly defined frequency
@trainer.on(Events.ITERATION_COMPLETED(event_filter=first_x_iters))
def log_gradients():
    # ...

```
</div>

---

### Power of Events & Handlers

#### 3. Custom events to go beyond standard events

<div style="font-size: 20px;">

```python
from ignite.engine import EventEnum

# Define custom events
class BackpropEvents(EventEnum):
    BACKWARD_STARTED = 'backward_started'
    BACKWARD_COMPLETED = 'backward_completed'
    OPTIM_STEP_COMPLETED = 'optim_step_completed'

def train_step(engine, batch):
    # ...
    loss = criterion(y_pred, y)
    engine.fire_event(BackpropEvents.BACKWARD_STARTED)
    loss.backward()
    engine.fire_event(BackpropEvents.BACKWARD_COMPLETED)
    optimizer.step()
    engine.fire_event(BackpropEvents.OPTIM_STEP_COMPLETED)
    # ...

trainer = Engine(train_step)
trainer.register_events(*BackpropEvents)

@trainer.on(BackpropEvents.BACKWARD_STARTED)
def function_before_backprop(engine):
    # ...
```
</div>

---

# Out-of-the-box metrics

50+ distributed ready out-of-the-box metrics to easily evaluate models.

- Dedicated to many Deep Learning tasks
- Easily composable to assemble a custom metric
- Easily extendable to create custom metrics

{{< add_vspace >}}

```python
precision = Precision(average=False)
recall = Recall(average=False)
F1_per_class = (precision * recall * 2 / (precision + recall))
F1_mean = F1_per_class.mean()  # torch mean method
F1_mean.attach(engine, "F1")
```

---

# Built-in Handlers

<table style="font-size: 20px;">
<tr>

<td>

{{< add_vspace >}}

- Logging to experiment tracking systems
- Checkpointing,
- Early stopping
- Profiling
- Parameter scheduling
- etc


</td>

<td>

```python
# model checkpoint handler
checkpoint = ModelChckpoint('/tmp/ckpts', 'training')
trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), handler, {'model': model})

# early stopping handler
def score_function(engine):
    val_loss = engine.state.metrics['acc']
    return val_loss
es = EarlyStopping(3, score_function, trainer)
evaluator.add_event_handler(Events.COMPLETED, handler)

# Piecewise linear parameter scheduler
scheduler = PiecewiseLinear(optimizer, 'lr', [(10, 0.5), (20, 0.45), (21, 0.3), (30, 0.1), (40, 0.1)])
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

# TensorBoard logger: batch loss, metrics
tb_logger = TensorboardLogger(log_dir="tb-logger")
tb_logger.attach_output_handler(
    trainer, event_name=Events.ITERATION_COMPLETED(every=100), tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

tb_logger.attach_output_handler(
    evaluator, event_name=Events.EPOCH_COMPLETED,
    tag="training", metric_names="all",
    global_step_transform=global_step_from_engine(trainer),
)
```

</td>

</tr>
</table>



---

# Distributed Training support

Run the same code across all supported backends seamlessly

- Backends from native torch distributed configuration: `nccl`, `gloo`, `mpi`
- Horovod framework with `gloo` or `nccl` communication backend
- XLA on TPUs via `pytorch/xla`

<div style="font-size: 20px;">

```python
import ignite.distributed as idist

def training(local_rank, *args, **kwargs):
    dataloder_train = idist.auto_dataloder(dataset, ...)

    model = ...
    model = idist.auto_model(model)

    optimizer = ...
    optimizer = idist.auto_optimizer(optimizer)

backend = 'nccl'  # or 'gloo', 'horovod', 'xla-tpu' or None
with idist.Parallel(backend) as parallel:
    parallel.run(training)
```
</div>

---

# Distributed Training support

## Distributed launchers

Handle distributed launchers with the same code
- `torch.multiprocessing.spawn`
- `torch.distributed.launch`
- `horovodrun`
- `slurm`

---

# Distributed Training support

## Unified Distributed API

- High-level helper methods
  - `idist.auto_model()`
  - `idist.auto_optim()`
  - `idist.auto_dataloader()`

- Collective operations
  - `all_reduce`, `all_gather`, and more

---

# The Big Picture

<img height="500" src="images/Ignite_Big_Picture.png"/>

<!-- End vertical slides -->
{{% /section %}}