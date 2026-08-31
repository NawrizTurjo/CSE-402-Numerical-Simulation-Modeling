"""Deterministic single-server simulation matching the slide example."""

interarrival_times = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
service_times = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]
customers_to_delay = 6

clock = 0
last_event_time = 0
next_arrival = interarrival_times[0]
next_departure = float("inf")

arrival_index = 1
service_index = 0
queue = []
server_busy = False
number_delayed = 0
next_customer_number = 1
customer_in_service = None
event_number = 0

total_delay = 0
queue_area = 0
busy_area = 0

print("e0: Initialization at time 0")
print("    Queue = 0, Server = idle, Number delayed = 0\n")

while number_delayed < customers_to_delay:
    # Select the next event.
    if next_arrival < next_departure:
        clock = next_arrival
        event = "arrival"
    else:
        clock = next_departure
        event = "departure"

    # Update time-average areas BEFORE changing the system state.
    time_since_last_event = clock - last_event_time
    queue_area += len(queue) * time_since_last_event

    if server_busy:
        busy_area += time_since_last_event

    last_event_time = clock
    event_number += 1

    if event == "arrival":
        customer = next_customer_number
        next_customer_number += 1
        old_queue_length = len(queue)

        # Schedule the following arrival.
        next_arrival = clock + interarrival_times[arrival_index]
        arrival_index += 1

        if server_busy:
            queue.append([customer, clock])
            print(
                f"e{event_number}: Arrival of C{customer} at time {clock:.1f} | "
                f"Queue {old_queue_length} -> {len(queue)}"
            )
        else:
            # This customer starts service immediately.
            server_busy = True
            number_delayed += 1
            customer_in_service = customer
            next_departure = clock + service_times[service_index]
            service_index += 1
            print(
                f"e{event_number}: Arrival of C{customer} at time {clock:.1f} | "
                "Service starts, delay = 0"
            )

    else:
        departed_customer = customer_in_service
        old_queue_length = len(queue)

        if len(queue) == 0:
            server_busy = False
            customer_in_service = None
            next_departure = float("inf")
            print(
                f"e{event_number}: Departure of C{departed_customer} "
                f"at time {clock:.1f} | Server becomes idle"
            )
        else:
            # The first waiting customer starts service.
            next_customer = queue.pop(0)
            customer_in_service = next_customer[0]
            customer_arrival_time = next_customer[1]
            delay = clock - customer_arrival_time
            total_delay += delay
            number_delayed += 1

            next_departure = clock + service_times[service_index]
            service_index += 1

            print(
                f"e{event_number}: Departure of C{departed_customer} "
                f"at time {clock:.1f} | Queue {old_queue_length} -> {len(queue)}"
            )
            print(
                f"    C{customer_in_service} starts service, "
                f"delay = {delay:.1f}"
            )

    print(
        f"    Queue = {len(queue)}, Number delayed = {number_delayed}, "
        f"Queue area = {queue_area:.1f}, Busy area = {busy_area:.1f}\n"
    )


average_delay = total_delay / customers_to_delay
time_average_queue = queue_area / clock
server_utilization = busy_area / clock

print("Simulation ending time:", round(clock, 2))
print("Total delay:", round(total_delay, 2))
print("Average delay in queue:", round(average_delay, 2))
print("Area under Q(t):", round(queue_area, 2))
print("Time-average number in queue:", round(time_average_queue, 2))
print("Area under B(t):", round(busy_area, 2))
print("Server utilization:", round(server_utilization, 2))
