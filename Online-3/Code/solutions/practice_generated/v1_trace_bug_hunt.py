"""
Practice V1: Verification by Trace -- Bug Hunt.
Demonstrates area-accumulation order in discrete-event simulation.
"""

import heapq


def simulate_queue_trace(is_buggy=False):
    interarrivals = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
    services = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]

    arrival_times = []
    t = 0.0
    for a in interarrivals:
        t += a
        arrival_times.append(t)

    events = [(at, i, "A", i + 1) for i, at in enumerate(arrival_times)]
    heapq.heapify(events)

    clock = 0.0
    last_time = 0.0
    busy = False
    queue = []
    area_Q = 0.0
    area_B = 0.0
    delays = []
    svc_iter = iter(services)
    seq = len(arrival_times)

    while events:
        clock, _, etype, cust = heapq.heappop(events)

        if not is_buggy:
            # CORRECT: accumulate area BEFORE state update
            dt = clock - last_time
            area_Q += len(queue) * dt
            area_B += (1.0 if busy else 0.0) * dt
            last_time = clock
        else:
            old_last_time = last_time

        # Update state
        if etype == "A":
            if not busy:
                busy = True
                delays.append(0.0)
                st = next(svc_iter, None)
                if st is not None:
                    heapq.heappush(events, (clock + st, seq, "D", cust))
                    seq += 1
            else:
                queue.append((cust, clock))
        else:  # Departure
            busy = False
            if queue:
                next_cust, arr_t = queue.pop(0)
                delays.append(clock - arr_t)
                busy = True
                st = next(svc_iter, None)
                if st is not None:
                    heapq.heappush(events, (clock + st, seq, "D", next_cust))
                    seq += 1

        if is_buggy:
            # BUGGY: accumulate area AFTER state update (uses wrong rectangle height)
            dt = clock - old_last_time
            area_Q += len(queue) * dt
            area_B += (1.0 if busy else 0.0) * dt
            last_time = clock

        if len(delays) == 6:
            break

    T = last_time
    avg_delay = sum(delays) / len(delays)
    avg_queue = area_Q / T
    utilization = area_B / T
    return avg_delay, avg_queue, utilization, T


def run_bug_hunt():
    print("--- Verification by Trace: Bug Hunt ---")
    d_bug, q_bug, u_bug, t_bug = simulate_queue_trace(is_buggy=True)
    d_fix, q_fix, u_fix, t_fix = simulate_queue_trace(is_buggy=False)

    print(f"BUGGY Simulation : Avg Delay = {d_bug:.4f}, Avg Queue = {q_bug:.4f}, Utilization = {u_bug:.4f}")
    print(f"FIXED Simulation : Avg Delay = {d_fix:.4f}, Avg Queue = {q_fix:.4f}, Utilization = {u_fix:.4f}")
    print(f"EXPECTED (Slide) : Avg Delay = 0.9500, Avg Queue = 1.1512, Utilization = 0.8953")
    print("\nLesson: Updating state variables BEFORE integrating area inflates the queue and utilization stats.")


run_bug_hunt()
