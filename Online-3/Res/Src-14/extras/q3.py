def single_server_queue_traced(interarrival_times, service_times, n_delays_to_observe):
    arrival_times, t = [], 0.0
    for a in interarrival_times:
        t += a
        arrival_times.append(t)

    events = []
    for i, at in enumerate(arrival_times):
        heapq.heappush(events, (at, 0, 'A', i + 1))

    clock = last_event_time = 0.0
    server_busy = False
    queue = []
    area_Q = area_B = 0.0
    delays = []
    svc_iter = iter(service_times)

    print(f"{'CLOCK':>6} {'EVENT':>10} {'#QUEUE':>7} {'STATUS':>8}")
    while events:
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

        label = f"Arr C{cust}" if etype == 'A' else f"Dep C{cust}"
        status = "Busy" if server_busy else "Idle"
        print(f"{clock:6.1f} {label:>10} {len(queue):7d} {status:>8}")

        if len(delays) == n_delays_to_observe:
            break

    return sum(delays)/len(delays), area_Q/last_event_time, area_B/last_event_time