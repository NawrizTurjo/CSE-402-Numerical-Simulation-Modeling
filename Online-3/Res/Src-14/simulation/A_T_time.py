import heapq
from queue import Queue

interarrival = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
service      = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]

def simulate_by_time(interarrival, service, T_max):
    event_list = []
    heapq.heappush(event_list, (interarrival[0], 'A'))
    arrival_list = Queue()

    interarrival_index = 1
    service_index = 0
    clock = last_event_time = 0.0
    is_busy = 0
    q_len = 0
    area_Q = area_B = 0.0
    delay = 0.0
    delay_count = 0

    while event_list:
        # STOP CONDITION: peek at next event without popping
        if event_list[0][0] > T_max:
            break

        clock, etype = heapq.heappop(event_list)

        dt = clock - last_event_time
        area_Q += q_len * dt       # rectangle before state changes
        area_B += is_busy * dt
        last_event_time = clock

        if etype == 'A':
            if is_busy == 0:
                is_busy = 1
                delay += 0
                delay_count += 1
                if service_index < len(service):
                    heapq.heappush(event_list, (clock + service[service_index], 'D'))
                    service_index += 1
                if interarrival_index < len(interarrival):
                    heapq.heappush(event_list, (clock + interarrival[interarrival_index], 'A'))
                    interarrival_index += 1
            else:
                q_len += 1
                arrival_list.put(clock)
                if interarrival_index < len(interarrival):
                    heapq.heappush(event_list, (clock + interarrival[interarrival_index], 'A'))
                    interarrival_index += 1
        else:
            if not arrival_list.empty():
                waited_since = arrival_list.get()
                delay += clock - waited_since
                delay_count += 1
            if q_len > 0:
                q_len -= 1
                if service_index < len(service):
                    heapq.heappush(event_list, (clock + service[service_index], 'D'))
                    service_index += 1
            else:
                is_busy = 0

    # TAIL: add the final rectangle from last event to T_max
    # State didn't change after the last processed event
    area_Q += q_len  * (T_max - last_event_time)
    area_B += is_busy * (T_max - last_event_time)

    avg_delay   = delay / delay_count if delay_count > 0 else 0.0
    avg_queue   = area_Q / T_max
    utilization = area_B / T_max
    return avg_delay, avg_queue, utilization

d, q, u = simulate_by_time(interarrival, service, T_max=8.6)
print(f"avg_delay={d:.4f}  avg_queue={q:.4f}  utiliz.={u:.4f}")
# avg_queue=1.1512  utiliz.=0.8953  ✅
# (avg_delay differs from Variant A because different customers completed by T=8.6)



def compute_performance(delays, event_times, Q_after, B_after):
    """
    Use when the exam gives you a trace table instead of asking you to simulate.

    delays      : list of each customer's delay (0.0 if server was idle on arrival)
    event_times : [t0=0.0, t1, t2, ..., tN=T]  — N+1 entries including start and end
    Q_after[i]  : queue length AFTER event i fires, held during [t_i, t_{i+1}]
    B_after[i]  : server status (0/1) AFTER event i fires, same interval
    """
    T = event_times[-1]

    # area = sum of rectangles: height × width for each interval
    area_Q = sum(Q_after[i] * (event_times[i+1] - event_times[i])
                 for i in range(len(Q_after)))
    area_B = sum(B_after[i] * (event_times[i+1] - event_times[i])
                 for i in range(len(B_after)))

    avg_delay   = sum(delays) / len(delays)
    avg_queue   = area_Q / T
    utilization = area_B / T
    return avg_delay, avg_queue, utilization


# Data from the slide's 13-event trace (seed: interarrival and service above)
delays      = [0.0, 0.8, 1.0, 0.0, 0.9, 3.0]   # D1..D6

# t0=0.0 (start), then each event time, ending at T=8.6
event_times = [0.0, 0.4, 1.6, 2.1, 2.4, 3.1, 3.3, 3.8, 4.0, 4.9, 5.6, 5.8, 7.2, 8.6]

# State AFTER each event fires (13 intervals)
# Event:   init  Arr1  Arr2  Arr3  Dep1  Dep2  Dep3  Arr4  Arr5  Dep4  Arr6  Arr7  Arr8
Q_after = [  0,    0,    1,    2,    1,    0,    0,    0,    1,    0,    1,    2,    3 ]
B_after = [  0,    1,    1,    1,    1,    1,    0,    1,    1,    1,    1,    1,    1 ]

d, q, u = compute_performance(delays, event_times, Q_after, B_after)
print(f"avg_delay={d:.4f}  avg_queue={q:.4f}  utiliz.={u:.4f}")
# avg_delay=0.9500  avg_queue=1.1512  utiliz.=0.8953  ✅




import heapq

interarrival = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
service = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]

T_MAX = 8.6   # stop after this much simulated time instead of after N customers


def simulate():
    event_list = [(interarrival[0], "A")]
    arrival_queue = []  # FIFO of arrival times waiting for service

    interarrival_index = 1
    service_index = 0

    q_len = 0
    is_busy = False
    last_event_time = 0

    area_queue = 0      # area under queue-length curve
    area_busy = 0        # area under server-busy curve
    total_delay = 0
    delay_count = 0

    while event_list:
        if event_list[0][0] > T_MAX:
            break

        curr_time, kind = heapq.heappop(event_list)
        print(f"current time: {curr_time}")

        elapsed = curr_time - last_event_time
        area_queue += q_len * elapsed
        area_busy += is_busy * elapsed

        if kind == "A":
            print("arrival")
            if not is_busy:
                is_busy = True
                delay_count += 1
                print(f"DELAY {delay_count} = 0")
                if service_index < len(service):
                    next_service_end = curr_time + service[service_index]
                    heapq.heappush(event_list, (next_service_end, "D"))
                    service_index += 1
                    print(f"next service end: {next_service_end}")
            else:
                q_len += 1
                arrival_queue.append(curr_time)

            if interarrival_index < len(interarrival):
                next_arrival = curr_time + interarrival[interarrival_index]
                heapq.heappush(event_list, (next_arrival, "A"))
                interarrival_index += 1
                print(f"next arrival time: {next_arrival}")

        else:
            print("departure")
            if arrival_queue:
                wait_started_at = arrival_queue.pop(0)
                delay = curr_time - wait_started_at
                total_delay += delay
                delay_count += 1
                print(f"DELAY {delay_count} = {delay}")

            if q_len > 0:
                q_len -= 1
                if service_index < len(service):
                    next_service_end = curr_time + service[service_index]
                    heapq.heappush(event_list, (next_service_end, "D"))
                    service_index += 1
                    print(f"next service end: {next_service_end}")
            else:
                is_busy = False

        last_event_time = curr_time

    # tail rectangle: state didn't change between last event and T_MAX
    area_queue += q_len * (T_MAX - last_event_time)
    area_busy += is_busy * (T_MAX - last_event_time)

    avg_delay = total_delay / delay_count if delay_count > 0 else 0
    avg_queue = area_queue / T_MAX
    utilization = area_busy / T_MAX
    print(avg_delay, avg_queue, utilization)
    return avg_delay, avg_queue, utilization


simulate()