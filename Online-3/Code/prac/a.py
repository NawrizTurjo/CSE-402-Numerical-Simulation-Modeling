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

interarrival = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
service = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]

r = simulate_ssq(interarrival, service, num_delays=6)
print(r)
r = simulate_ssq(interarrival, service, t_max=8.6)
print(r)
interarrival_drain = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6]
service_drain = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]
r = simulate_ssq(interarrival_drain, service_drain, trace=True)
print(r)
            