import heapq
from queue import Queue

interarrival = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
service      = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]

def simulate_with_trace(interarrival, service, n_delays=6):
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

    # Print trace header
    print(f"{'CLOCK':>7} {'EVENT':>8} {'Q':>4} {'SERVER':>8} {'DELAY':>7}")
    print("-" * 40)

    while event_list:
        clock, etype = heapq.heappop(event_list)
        dt = clock - last_event_time
        area_Q += q_len * dt
        area_B += is_busy * dt
        last_event_time = clock

        if etype == 'A':
            if is_busy == 0:
                is_busy = 1
                delay += 0
                delay_count += 1
                status = "Busy"
                d_str = "0.0"
                if service_index < len(service):
                    heapq.heappush(event_list, (clock + service[service_index], 'D'))
                    service_index += 1
                if interarrival_index < len(interarrival):
                    heapq.heappush(event_list, (clock + interarrival[interarrival_index], 'A'))
                    interarrival_index += 1
            else:
                q_len += 1
                arrival_list.put(clock)
                status = "Busy"
                d_str = "-"
                if interarrival_index < len(interarrival):
                    heapq.heappush(event_list, (clock + interarrival[interarrival_index], 'A'))
                    interarrival_index += 1
        else:
            if not arrival_list.empty():
                waited_since = arrival_list.get()
                d = clock - waited_since
                delay += d
                delay_count += 1
                d_str = f"{d:.1f}"
            else:
                d_str = "-"
            if q_len > 0:
                q_len -= 1
                status = "Busy"
                if service_index < len(service):
                    heapq.heappush(event_list, (clock + service[service_index], 'D'))
                    service_index += 1
            else:
                is_busy = 0
                status = "Idle"
                if interarrival_index < len(interarrival):
                    heapq.heappush(event_list, (1e9, 'D'))

        # Print one trace row per event
        if clock < 1e8:
            print(f"{clock:>7.1f} {etype:>8} {q_len:>4} {status:>8} {d_str:>7}")

        if delay_count == n_delays:
            T = last_event_time
            print("-" * 40)
            print(f"\nFinal Results (T={T}):")
            print(f"  avg_delay   = {delay/delay_count:.4f}  (expected 0.95)")
            print(f"  avg_queue   = {area_Q/T:.4f}  (expected 1.15)")
            print(f"  utilization = {area_B/T:.4f}  (expected 0.90)")
            return

simulate_with_trace(interarrival, service)