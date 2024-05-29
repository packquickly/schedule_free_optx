# Schedule-free optimisers in JAX using Optimistix

An implementation of the new [schedule-free SGD and AdamW](https://arxiv.org/abs/2405.15682) optimisers in JAX using [Optimistix](https://github.com/patrick-kidger/optimistix).

<img width="430" alt="Screenshot 2024-05-29 at 10 24 28 PM" src="https://github.com/packquickly/schedule_free_optx/assets/38091354/10cd742e-be8d-43c6-b7ce-1b982c791047">

# Installation

To install, simply download this repo and run

```pip install -e .```

in your Python environment. Then use `from schedule_free_optx import ScheduleFreeSGD, ScheduleFreeAdamW` and use them as any other Optimistix optimiser!
