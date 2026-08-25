"""
Single-Server Queue Discrete-Event Simulation
Next-Event Time Advance algorithm (Law & Kelton style)

State variables:
    clock              -> simulation clock
    server_status      -> 0 = idle, 1 = busy   -> represents B(t)
    num_in_queue       -> number of customers waiting (not in service) -> represents Q(t)

Time-weighted (area) accumulators:
    area_Q  -> integral of Q(t) dt   (used for time-average number in queue)
    area_B  -> integral of B(t) dt   (used for server utilization)

Customer-average accumulators:
    total_delay          -> sum of waiting time in queue for every customer
    num_customers_served -> count of customers who have completed service
"""

import math
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. INPUT DATA: inter-arrival times and service times for each customer
#    (In a real simulation these would usually be random draws; here we
#     use fixed arrays so the logic/output is fully reproducible.)
# ---------------------------------------------------------------------
inter_arrival_times = [2, 3, 1, 4, 2, 5, 1, 3]   # time between successive arrivals
service_times       = [4, 2, 3, 1, 2, 3, 2, 4]   # service time required by each customer

N = min(len(inter_arrival_times), len(service_times))  # number of customers to simulate

INFINITY = math.inf

# ---------------------------------------------------------------------
# 2. INITIALIZATION
# ---------------------------------------------------------------------
clock = 0.0
time_last_event = 0.0

server_status = 0        # 0 = idle, 1 = busy  -> B(t)
num_in_queue = 0          # -> Q(t)
queue_arrival_times = []  # arrival times of customers currently waiting (FIFO)

area_Q = 0.0   # cumulative area under Q(t) curve
area_B = 0.0   # cumulative area under B(t) curve

# Lists to record state transitions over time for plotting Q(t) and B(t)
time_history = [0.0]
q_history = [0]
b_history = [0]

total_delay = 0.0
num_customers_served = 0
num_customers_delayed = 0   # customers who actually had to wait (for optional stats)

arrival_index = 0          # index into inter_arrival_times / who arrives next
service_index = 0          # index into service_times / whose service time is next used

next_arrival_time = inter_arrival_times[0]   # first arrival
next_departure_time = INFINITY               # no one in service yet

print(f"{'Time':>7} | {'Event':<12} | {'Q(t)':>4} | {'B(t)':>4} | Details")
print("-" * 70)

# ---------------------------------------------------------------------
# 3. MAIN EVENT LOOP  (Next-Event Time Advance)
#    Stop when all customers have arrived AND the system is empty
#    (no one waiting, server idle).
# ---------------------------------------------------------------------
while arrival_index < N or server_status == 1 or num_in_queue > 0:

    # ---- determine next event: whichever is sooner ----
    next_event_time = min(next_arrival_time, next_departure_time)

    # ---- advance time-weighted statistics up to next_event_time ----
    elapsed = next_event_time - time_last_event
    area_Q += num_in_queue * elapsed
    area_B += server_status * elapsed
    time_last_event = next_event_time

    # ---- advance the clock ----
    clock = next_event_time

    # =========================== ARRIVAL EVENT ===========================
    if next_arrival_time <= next_departure_time:

        if server_status == 1:
            # server busy -> customer joins the queue
            num_in_queue += 1
            queue_arrival_times.append(clock)
            print(f"{clock:7.2f} | {'ARRIVAL':<12} | {num_in_queue:4d} | {server_status:4d} | "
                  f"Customer {arrival_index + 1} arrives, server busy, joins queue")
        else:
            # server idle -> customer starts service immediately, no delay
            server_status = 1
            num_customers_served += 1
            svc = service_times[service_index]
            service_index += 1
            next_departure_time = clock + svc
            print(f"{clock:7.2f} | {'ARRIVAL':<12} | {num_in_queue:4d} | {server_status:4d} | "
                  f"Customer {arrival_index + 1} arrives, starts service immediately "
                  f"(delay=0.00, service time={svc})")

        arrival_index += 1
        if arrival_index < N:
            next_arrival_time = clock + inter_arrival_times[arrival_index]
        else:
            next_arrival_time = INFINITY   # no more arrivals scheduled

    # ========================== DEPARTURE EVENT ===========================
    else:
        if num_in_queue == 0:
            # queue empty -> server becomes idle
            server_status = 0
            next_departure_time = INFINITY
            print(f"{clock:7.2f} | {'DEPARTURE':<12} | {num_in_queue:4d} | {server_status:4d} | "
                  f"A customer departs, queue empty, server goes idle")
        else:
            # pull next customer from the queue into service
            num_in_queue -= 1
            cust_arrival = queue_arrival_times.pop(0)
            delay = clock - cust_arrival
            total_delay += delay
            num_customers_delayed += 1
            num_customers_served += 1

            svc = service_times[service_index]
            service_index += 1
            next_departure_time = clock + svc

            print(f"{clock:7.2f} | {'DEPARTURE':<12} | {num_in_queue:4d} | {server_status:4d} | "
                  f"A customer departs, next customer starts service "
                  f"(delay={delay:.2f}, service time={svc})")

    # Record state transition for plotting
    time_history.append(clock)
    q_history.append(num_in_queue)
    b_history.append(server_status)

# ---------------------------------------------------------------------
# 4. FINAL REPORT
# ---------------------------------------------------------------------
print("-" * 70)
print(f"Simulation ended at time {clock:.2f}")
print(f"Total customers processed        : {num_customers_served}")

avg_num_in_queue = area_Q / clock
server_utilization = area_B / clock
avg_delay = total_delay / N   # average delay over ALL customers (0 counted for those not delayed)

print(f"Time-average number in queue Q̄  : {avg_num_in_queue:.4f}")
print(f"Server utilization (rho)         : {server_utilization:.4f}")
print(f"Average delay in queue per cust. : {avg_delay:.4f}")

# ---------------------------------------------------------------------
# 5. PLOTTING Q(t) and B(t)
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Q(t) vs Time
ax1.step(time_history, q_history, where='post', color='royalblue', linewidth=2, label='Q(t)')
ax1.fill_between(time_history, q_history, step='post', alpha=0.2, color='royalblue')
ax1.set_ylabel('Number in Queue Q(t)', fontsize=11)
ax1.set_title('Single-Server Queue: Q(t) and B(t) vs Time', fontsize=13, fontweight='bold')
ax1.set_yticks(range(max(q_history) + 2))
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper right')

# B(t) vs Time
ax2.step(time_history, b_history, where='post', color='darkorange', linewidth=2, label='B(t)')
ax2.fill_between(time_history, b_history, step='post', alpha=0.2, color='darkorange')
ax2.set_xlabel('Simulation Time (t)', fontsize=11)
ax2.set_ylabel('Server Status B(t)', fontsize=11)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['0 (Idle)', '1 (Busy)'])
ax2.set_ylim(-0.1, 1.2)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()