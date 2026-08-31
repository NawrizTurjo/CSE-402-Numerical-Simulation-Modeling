"""
Practice S1: Coffee Shop Queue Simulation.
Single/multi-server discrete-event queue simulation.
"""

import heapq


def simulate_coffee_shop(num_servers=1, capacity=None, num_delays=8):
    interarrivals = [1.5, 0.8, 2.1, 0.3, 1.2, 0.9, 1.7, 0.4, 2.3, 0.6]
    services = [2.5, 1.1, 0.9, 1.8, 0.7, 2.0, 1.3, 0.85]

    # Precompute arrival times
    arrival_times = []
    t = 0.0
    for a in interarrivals:
        t += a
        arrival_times.append(t)

    events = [(at, i, "A", i + 1) for i, at in enumerate(arrival_times)]
    heapq.heapify(events)

    clock = 0.0
    last_time = 0.0
    busy_servers = 0
    queue = []
    area_Q = 0.0
    area_B = 0.0
    delays = []
    balked = 0
    svc_iter = iter(services)
    seq = len(arrival_times)

    while events:
        clock, _, etype, cust = heapq.heappop(events)

        # 1. Accumulate areas using OLD state first
        dt = clock - last_time
        area_Q += len(queue) * dt
        area_B += busy_servers * dt
        last_time = clock

        # 2. Update state according to event
        if etype == "A":
            if busy_servers < num_servers:
                busy_servers += 1
                delays.append(0.0)
                st = next(svc_iter, None)
                if st is not None:
                    heapq.heappush(events, (clock + st, seq, "D", cust))
                    seq += 1
            elif capacity is not None and len(queue) >= capacity:
                balked += 1
            else:
                queue.append((cust, clock))
        else:  # Departure
            busy_servers -= 1
            if queue:
                next_cust, arr_t = queue.pop(0)
                delays.append(clock - arr_t)
                busy_servers += 1
                st = next(svc_iter, None)
                if st is not None:
                    heapq.heappush(events, (clock + st, seq, "D", next_cust))
                    seq += 1

        if len(delays) == num_delays:
            break

    T = last_time
    avg_delay = sum(delays) / len(delays)
    avg_queue = area_Q / T
    utilization = area_B / (num_servers * T)

    return avg_delay, avg_queue, utilization, T, balked


def run_coffee_shop():
    print("--- Coffee Shop Queue Simulation ---")

    # 1 Barista
    d1, q1, u1, t1, _ = simulate_coffee_shop(num_servers=1)
    print(f"1 Barista  : Avg Delay = {d1:.2f} min, Avg Queue = {q1:.2f}, Utilization = {u1:.2%}")

    # 2 Baristas
    d2, q2, u2, t2, _ = simulate_coffee_shop(num_servers=2)
    print(f"2 Baristas : Avg Delay = {d2:.2f} min, Avg Queue = {q2:.2f}, Utilization = {u2:.2%}")

    # Balking (waiting capacity = 2)
    d3, q3, u3, t3, lost = simulate_coffee_shop(num_servers=1, capacity=2)
    print(f"Balking (k=2): Avg Delay = {d3:.2f} min, Customers Lost = {lost}")


run_coffee_shop()
