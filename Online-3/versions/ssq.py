import heapq


def single_server_queue(interarrival_times, service_times, n_delays_to_observe):
    arrival_times, t = [], 0.0
    for a in interarrival_times:
        t += a
        arrival_times.append(t)

    events = []
    for i, at in enumerate(arrival_times):
        heapq.heappush(events, (at, 0, "A", i + 1))

    clock = last_event_time = 0.0
    server_busy = False
    queue = []
    area_Q = area_B = 0.0
    delays = []
    svc_iter = iter(service_times)

    while events:
        clock, _, etype, cust = heapq.heappop(events)

        dt = clock - last_event_time
        area_Q += len(queue) * dt
        area_B += (1.0 if server_busy else 0.0) * dt
        last_event_time = clock

        if etype == "A":
            if not server_busy:
                server_busy = True
                delays.append(0.0)
                st = next(svc_iter)
                heapq.heappush(events, (clock + st, 1, "D", cust))
            else:
                queue.append((cust, clock))
        else:
            if queue:
                next_cust, arr_t = queue.pop(0)
                delays.append(clock - arr_t)
                st = next(svc_iter)
                heapq.heappush(events, (clock + st, 1, "D", next_cust))
            else:
                server_busy = False

        if len(delays) == n_delays_to_observe:
            break

    print(f"server utilization: {area_B / last_event_time:.4f}")
    print(f"average number in queue: {area_Q / last_event_time:.4f}")

    T = last_event_time
    avg_delay = sum(delays) / len(delays)
    avg_num_in_queue = area_Q / T
    utilization = area_B / T
    return avg_delay, avg_num_in_queue, utilization, T

A = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
S = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]

sim_results = single_server_queue(A, S, n_delays_to_observe=6)
print(f"Average delay: {sim_results[0]:.4f}")
print(f"Average number in queue: {sim_results[1]:.4f}")
print(f"Server utilization: {sim_results[2]:.4f}")
print(f"Total simulation time: {sim_results[3]:.4f}")