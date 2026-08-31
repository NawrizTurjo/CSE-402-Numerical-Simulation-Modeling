"""Deterministic single-server simulation using user input."""

interarrival_input = input("Enter interarrival times: ").replace(",", " ")
service_input = input("Enter service times: ").replace(",", " ")

interarrival_times = list(map(float, interarrival_input.split()))
service_times = list(map(float, service_input.split()))

arrival_time = 0
finish_time = 0
total_delay = 0
total_service_time = 0

for interarrival, service in zip(interarrival_times, service_times):
    arrival_time += interarrival
    service_start = max(arrival_time, finish_time)
    total_delay += service_start - arrival_time
    total_service_time += service
    finish_time = service_start + service

customers = len(service_times)
average_delay = total_delay / customers
utilization = total_service_time / finish_time

print("Average delay:", round(average_delay, 2))
print("Server utilization:", round(utilization, 2))
print("Ending time:", round(finish_time, 2))
