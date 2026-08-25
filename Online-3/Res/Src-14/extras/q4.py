def multi_server_queue(interarrival_times, service_times, c, n_delays_to_observe):
    arrival_times, t = [], 0.0
    for a in interarrival_times:
        t += a
        arrival_times.append(t)

    events = []
    for i, at in enumerate(arrival_times):
        heapq.heappush(events, (at, 0, 'A', i + 1))

    clock = last_event_time = 0.0
    busy_servers = 0
    queue = []
    area_Q = area_B = 0.0
    delays = []
    svc_iter = iter(service_times)

    while events:
        clock, _, etype, cust = heapq.heappop(events)
        dt = clock - last_event_time
        area_Q += len(queue) * dt
        area_B += busy_servers * dt          # sum of busy servers over time
        last_event_time = clock

        if etype == 'A':
            if busy_servers < c:
                busy_servers += 1
                delays.append(0.0)
                st = next(svc_iter)
                heapq.heappush(events, (clock + st, 1, 'D', cust))
            else:
                queue.append((cust, clock))
        else:  # Departure -> one server freed
            busy_servers -= 1
            if queue:
                next_cust, arr_t = queue.pop(0)
                delays.append(clock - arr_t)
                busy_servers += 1
                st = next(svc_iter)
                heapq.heappush(events, (clock + st, 1, 'D', next_cust))

        if len(delays) == n_delays_to_observe:
            break

    T = last_event_time
    return sum(delays)/len(delays), area_Q/T, area_B/(c*T)   # utilization normalized by c servers