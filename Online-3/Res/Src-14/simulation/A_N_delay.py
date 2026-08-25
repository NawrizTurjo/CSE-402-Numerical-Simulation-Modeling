import heapq
from queue import Queue

interarrival = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
service      = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]

def simulate(interarrival, service, n_delays=6):
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
        clock, etype = heapq.heappop(event_list)

        # STEP 1: update areas using OLD state
        dt = clock - last_event_time
        area_Q += q_len * dt
        area_B += is_busy * dt
        last_event_time = clock

        if etype == 'A':                         # ARRIVAL EVENT
            if is_busy == 0:
                is_busy = 1
                delay += 0                       # zero delay — served immediately
                delay_count += 1
                if service_index < len(service):
                    heapq.heappush(event_list, (clock + service[service_index], 'D'))
                    service_index += 1
                if interarrival_index < len(interarrival):
                    heapq.heappush(event_list, (clock + interarrival[interarrival_index], 'A'))
                    interarrival_index += 1
            else:
                q_len += 1                       # joins the queue
                arrival_list.put(clock)
                if interarrival_index < len(interarrival):
                    heapq.heappush(event_list, (clock + interarrival[interarrival_index], 'A'))
                    interarrival_index += 1
        else:                                    # DEPARTURE EVENT
            if not arrival_list.empty():
                waited_since = arrival_list.get()
                delay += clock - waited_since    # real delay
                delay_count += 1
            if q_len > 0:
                q_len -= 1
                if service_index < len(service):
                    heapq.heappush(event_list, (clock + service[service_index], 'D'))
                    service_index += 1
            else:
                is_busy = 0
                if interarrival_index < len(interarrival):
                    heapq.heappush(event_list, (1e9, 'D'))  # keep heap alive

        if delay_count == n_delays:              # STOPPING CONDITION
            T = last_event_time
            return delay/delay_count, area_Q/T, area_B/T, T

d, q, u, T = simulate(interarrival, service, n_delays=6)
print(f"avg_delay={d:.4f}")    # 0.9500
print(f"avg_queue={q:.4f}")    # 1.1512
print(f"utiliz.  ={u:.4f}")    # 0.8953
print(f"T        ={T:.1f}")    # 8.6