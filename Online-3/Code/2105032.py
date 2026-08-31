from scipy import stats
import math
import random

import heapq
import itertools
import math

def simulate_ssq(
    interarrival_times,
    service_times,
    *,
    num_delays=None,
    t_max=None,
    num_servers=1,
    capacity=None,
    trace=False
):
    arrival_times = []
    t = 0.0
    for gap in interarrival_times:
        t+=gap
        arrival_times.append(t)

    seq = itertools.count()
    events = []

    for i, arrival_time in enumerate(arrival_times):
        if t_max is not None and arrival_time>t_max:
            break
        heapq.heappush(
            events,
            (
                arrival_time,
                next(seq),
                "A",
                i+1
            )
        )

    clock = last_event = 0.0
    busy_servers = 0
    queue = [] # (customer_id, arrival_time)
    area_Q = area_B = 0.0
    delays = []
    num_lost = 0
    service_iter = iter(service_times)
    trace_rows = []

    stop_reason = "drained"

    while events:
        if t_max is not None and events[0][0] > t_max:
            stop_reason = "t_max"
            break

        clock, _, event_type, customer_no = heapq.heappop(events)

        dt = clock - last_event
        area_Q += len(queue) * dt
        area_B += busy_servers * dt
        last_event = clock

        if event_type == "A":
            if busy_servers < num_servers:
                busy_servers+=1
                delays.append(0.0) # free server
                service_time = next(service_iter, None)
                if service_time is not None:
                    heapq.heappush(
                        events,
                        (
                            clock+service_time,
                            next(seq),
                            "D",
                            customer_no
                        )
                    )
                event_description = f"Arrival C{customer_no} (starts service, D=0)"
            elif capacity is not None and len(queue) >= capacity:
                num_lost+=1
                event_description = f"Arrival C{customer_no} (BALKS - queue full)"
            else:
                queue.append((
                    customer_no,
                    clock
                ))
                event_description = f"Arrival C{customer_no} (joins queue)"
        else:
            busy_servers-=1
            if queue:
                next_customer, arrival_time = queue.pop(0)
                delay = clock - arrival_time
                delays.append(delay)
                busy_servers+=1
                service_time = next(
                    service_iter, None
                )
                if service_time is not None:
                    heapq.heappush(
                        events,
                        (
                            clock+service_time,
                            next(seq),
                            "D",
                            next_customer
                        )
                    )
                event_description = f"Departure C{customer_no} (C{next_customer} starts service, D={delay:.2f})"
            else:
                event_description = f"Departure C{customer_no} (service goes idle)"
        if trace:
            trace_rows.append(
                {
                    "clock": clock,
                    "event": event_description,
                    "queue_len": len(queue),
                    "busy_servers": busy_servers,
                    "area_Q":area_Q,
                    "area_B":area_B
                }
            )
        if num_delays is not None and len(delays) == num_delays:
            stop_reason = "num_delays"
            break
    if stop_reason == "t_max":
        dt = t_max - last_event
        area_Q += len(queue) * dt
        area_B += busy_servers * dt
        T = t_max
    else:
        T = last_event

    result = {
        "avg_delay": sum(delays) / len(delays) if delays else 0.0,
        "avg_queue": area_Q / T if T else 0.0,
        "utilization": area_B / (num_servers * T) if T else 0.0,
        "T": T,
        "delays": delays,
        "num_lost": num_lost,
        "num_customers_delayed": len(delays),
        "stop_reason": stop_reason,
    }

    return result


def find_cycle(step, seed):

    # step is a function

    seen_at = {}
    curr = seed
    iteration = 0

    while curr not in seen_at:
        seen_at[curr] = iteration
        curr = step(curr)
        iteration+=1

    first_seen_iteration = seen_at[curr]
    cycle_len = iteration - first_seen_iteration
    return curr,iteration,cycle_len

def bin_counts(uniforms, bins=10):
    counts = [0] * bins
    for value in uniforms:
        index = min(
            bins-1, int(value*bins)
        ) # atmost bins-1 porjonto hobe
        counts[index] +=1
    return counts

# ============================randu==============================


def randu_step(x, a, c, m):
    return (a * x + c) % m

def randu(n,seed=1, a=65539, c=0, m=2**31):
    """Generate n successive integer states X1..Xn."""
    values = []
    x = seed
    for _ in range(n):
        x = randu_step(x, a, c, m)
        values.append(x)
    return values

def randu_uniforms(n,seed=1, a=65539, c=0, m=2**31):
    """Same sequence as randu(), rescaled to floats in [0, 1)."""
    return [x / m for x in randu(n,seed, a, c, m)]

def find_randu_cycle(seed=1, a=65539, c=0, m=2**31):
    return find_cycle(
        step=lambda x: randu_step(x,a,c,m),
        seed=seed
    )


samples = randu_uniforms(10)
print(samples)

def exponential(r, theta):
    """F(x) = 1 - e^(-x/mean) => x = -mean * ln(1 - r)"""
    return -1.0 * (math.log(1 - r) / theta)

# print(find_randu_cycle())

total_num = 20000
wm = 2000

customers = total_num-wm
samples = randu_uniforms(3 * total_num)
print(3 * (total_num-wm))
U = samples

C_k_1 = 0.0
A_k = 0.0
total_serv = 0.0
total_mean = 0.0
C, A = 0.0, 0.0

arrivals = []
service = []

for k in range(total_num):
    u_arr = U[3*k]
    u_serv = U[(3*k+2)]

    arr_k = exponential(u_arr, 3)    # interarrival gap
    arrivals.append(arr_k)
    
    srv_k = exponential(u_serv, 4)
    service.append(srv_k)

    A_k += arr_k                     # cumulative arrival time
    S_k = max(A_k, C_k_1)
    C_k_1 = S_k + srv_k

    if k >= 2000:
        total_mean += (C_k_1 - A_k)
        total_serv += srv_k

    if k == 19999:
        C = C_k_1
    if k == 2000:
        A = A_k

T = C - A
print(T)
rho = total_serv / T
print(rho)
L_hat = total_mean / T
print(L_hat)

arrivals = arrivals[wm:]
service = service[wm:]

# interarrival = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
# service = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]

r = simulate_ssq(arrivals, service,trace=False)
print(f"avg_delay={r['avg_delay']:.4f} avg_queue={r['avg_queue']:.4f} "
      f"utilization={r['utilization']:.4f} T={r['T']:.1f}")