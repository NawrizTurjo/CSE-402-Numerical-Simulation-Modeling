"""Simple deterministic simulation of one server and one waiting line."""

interarrival_times = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6]
service_times = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]

arrival_time = 0
previous_finish_time = 0
total_delay = 0
total_service_time = 0

for interarrival, service in zip(interarrival_times, service_times):
    arrival_time += interarrival

    # Service begins at arrival if the server is free; otherwise, the customer waits.
    service_start_time = max(arrival_time, previous_finish_time)
    delay = service_start_time - arrival_time
    finish_time = service_start_time + service

    total_delay += delay
    total_service_time += service
    previous_finish_time = finish_time

number_of_customers = len(service_times)
average_delay = total_delay / number_of_customers
server_utilization = total_service_time / previous_finish_time

print(f"Customers served: {number_of_customers}")
print(f"Average delay: {average_delay:.2f}")
print(f"Server utilization: {server_utilization:.2%}")
print(f"Simulation ended at: {previous_finish_time:.2f}")
