def single_server_queue_by_time(interarrival_times, service_times, T_max):
    arrival_times, t = [], 0.0
    for a in interarrival_times:
        t += a
        arrival_times.append(t)

    events = []
    for i, at in enumerate(arrival_times):
        if at <= T_max:
            heapq.heappush(events, (at, 0, 'A', i + 1))

    clock = last_event_time = 0.0
    server_busy = False
    queue = []
    area_Q = area_B = 0.0
    delays = []
    svc_iter = iter(service_times)

    while events and events[0][0] <= T_max:
        clock, _, etype, cust = heapq.heappop(events)
        dt = clock - last_event_time
        area_Q += len(queue) * dt
        area_B += (1.0 if server_busy else 0.0) * dt
        last_event_time = clock

        if etype == 'A':
            if not server_busy:
                server_busy = True
                delays.append(0.0)
                st = next(svc_iter)
                heapq.heappush(events, (clock + st, 1, 'D', cust))
            else:
                queue.append((cust, clock))
        else:
            if queue:
                next_cust, arr_t = queue.pop(0)
                delays.append(clock - arr_t)
                st = next(svc_iter)
                heapq.heappush(events, (clock + st, 1, 'D', next_cust))
            else:
                server_busy = False

    # account for the tail: from last_event_time to T_max, state doesn't change
    dt = T_max - last_event_time
    area_Q += len(queue) * dt
    area_B += (1.0 if server_busy else 0.0) * dt

    return sum(delays)/len(delays) if delays else 0, area_Q/T_max, area_B/T_max