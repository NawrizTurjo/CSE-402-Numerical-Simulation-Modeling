"""Easy stochastic single-server queue simulation (FCFS)."""

import random

random.seed(42)
number_of_customers = 10
mean_interarrival = 1.0
mean_service = 0.5

arrival_time = 0
previous_finish = 0
total_waiting = 0
total_service = 0

print("Customer  Arrival  Service  Start  Finish  Waiting")

for customer in range(1, number_of_customers + 1):
    interarrival = random.expovariate(1 / mean_interarrival)
    service_time = random.expovariate(1 / mean_service)
    arrival_time += interarrival

    start_time = max(arrival_time, previous_finish)
    finish_time = start_time + service_time
    waiting_time = start_time - arrival_time

    total_waiting += waiting_time
    total_service += service_time
    previous_finish = finish_time

    print(
        f"{customer:^8} {arrival_time:^8.2f} {service_time:^8.2f} "
        f"{start_time:^7.2f} {finish_time:^8.2f} {waiting_time:^7.2f}"
    )

average_waiting = total_waiting / number_of_customers
utilization = total_service / previous_finish

print("\nAverage waiting time:", round(average_waiting, 2))
print("Server utilization:", round(utilization * 100, 2), "%")
