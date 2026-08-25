import heapq

def single_server_queue(interarrival_times, service_times, n_delays_to_observe):
    """
    Next-event time-advance simulation of a single-server FIFO queue.
    Stops the instant the n-th delay is OBSERVED, i.e. when the n-th
    customer BEGINS service (not when they depart!).
    """
    # Precompute arrival times = cumulative sum of interarrival times
    arrival_times, t = [], 0.0
    for a in interarrival_times:
        t += a
        arrival_times.append(t)

    # Event list: (time, tie_breaker, type, customer_id)
    events = []
    for i, at in enumerate(arrival_times):
        heapq.heappush(events, (at, 0, 'A', i + 1))   # 'A' = arrival

    clock = last_event_time = 0.0
    server_busy = False
    queue = []              # FIFO list of (customer_id, arrival_time)
    area_Q = area_B = 0.0   # area under Q(t) and B(t)
    delays = []
    svc_iter = iter(service_times)

    while events:
        clock, _, etype, cust = heapq.heappop(events)

        # STEP 1: update area accumulators using OLD state, over the interval that just ended
        dt = clock - last_event_time
        area_Q += len(queue) * dt
        area_B += (1.0 if server_busy else 0.0) * dt
        last_event_time = clock   # STEP 3 (done early here for convenience)

        # STEP 2 + 4: update state & schedule new events
        if etype == 'A':
            if not server_busy:
                server_busy = True
                delays.append(0.0)                       # zero wait, served immediately
                st = next(svc_iter)
                heapq.heappush(events, (clock + st, 1, 'D', cust))
            else:
                queue.append((cust, clock))               # joins the queue
        else:  # Departure
            if queue:
                next_cust, arr_t = queue.pop(0)
                delays.append(clock - arr_t)               # delay ends when service STARTS
                st = next(svc_iter)
                heapq.heappush(events, (clock + st, 1, 'D', next_cust))
            else:
                server_busy = False                        # server goes idle

        if len(delays) == n_delays_to_observe:
            break   # stop as soon as the n-th delay has been observed

    T = last_event_time
    avg_delay = sum(delays) / len(delays)
    avg_num_in_queue = area_Q / T
    utilization = area_B / T
    return avg_delay, avg_num_in_queue, utilization, T


if __name__ == "__main__":
    interarrival = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
    service      = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]

    d, q, u, T = single_server_queue(interarrival, service, n_delays_to_observe=6)
    print(f"Average delay in queue : {d:.4f}")   # 0.9500
    print(f"Avg # in queue         : {q:.4f}")   # 1.1512 ≈ 1.15
    print(f"Server utilization     : {u:.4f}")   # 0.8953 ≈ 0.90
    print(f"Total sim time (T)     : {T}")        # 8.6