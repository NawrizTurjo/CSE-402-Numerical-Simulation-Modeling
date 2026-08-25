# Queue Simulation

## Basic Idea

Queue simulation models a waiting line with arrivals and services.

Example:

```text
Customers arrive at a server.
If the server is idle, service starts immediately.
If the server is busy, the customer waits in a queue.
```

Important quantities:

```text
delay              = waiting time before service starts
queue length       = number of customers waiting
server utilization = fraction of time server is busy
simulation clock   = current simulation time
```

## Next-Event Simulation

This project uses next-event time advance.

Instead of increasing time by tiny steps, the clock jumps directly to the next event.

Events:

```text
A = arrival
D = departure
```

The event list is stored using a priority queue:

```python
heapq.heappush(event_list, (time, event_type))
heapq.heappop(event_list)
```

The earliest event is processed first.

## Area Calculations

Average queue length is found using area under the queue-length curve:

```text
Average queue length = area_Q / total_time
```

Server utilization:

```text
Utilization = area_B / total_time
```

In code:

```python
dt = clock - last_event_time
area_Q += q_len * dt
area_B += is_busy * dt
```

This uses the old state during the time interval before the new event happens.

## Files

### `A_N_delay.py`

Simulates until a fixed number of customer delays has been observed.

Stopping condition:

```python
if delay_count == n_delays:
    return delay/delay_count, area_Q/T, area_B/T, T
```

This is useful when the question says to simulate until `N` customers begin service.

### `A_T_time.py`

Simulates until a fixed time `T_max`.

Stopping condition:

```python
if event_list[0][0] > T_max:
    break
```

It also adds the final rectangle from the last event time to `T_max`:

```python
area_Q += q_len * (T_max - last_event_time)
area_B += is_busy * (T_max - last_event_time)
```

### `simulation_with_trace.py`

Prints a trace table while simulating.

The trace shows:

```text
clock time
event type
queue length
server status
delay
```

This is helpful for matching class slides or exam trace tables.

### `variant_c.py`

Shows direct calculation from a given trace table.

It uses:

```python
area_Q = sum(Q_after[i] * (event_times[i+1] - event_times[i]) for i in range(len(Q_after)))
area_B = sum(B_after[i] * (event_times[i+1] - event_times[i]) for i in range(len(B_after)))
```

This is useful if the exam gives you event times and states instead of asking you to simulate from scratch.

## Run

```bash
python simulation/A_N_delay.py
python simulation/A_T_time.py
python simulation/simulation_with_trace.py
```
