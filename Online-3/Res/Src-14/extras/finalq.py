import heapq

interarrival = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
service = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]

NUM_CUSTOMERS = 6


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

    while event_list and delay_count < NUM_CUSTOMERS:
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

    avg_delay = total_delay / delay_count
    avg_queue = area_queue / last_event_time
    utilization = area_busy / last_event_time
    print(avg_delay, avg_queue, utilization)
    return avg_delay, avg_queue, utilization


simulate()
