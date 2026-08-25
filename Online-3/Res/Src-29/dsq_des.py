"""
Double-Server (2 parallel identical servers, single shared queue)
Discrete-Event Simulation using Next-Event Time Advance

State variables:
    clock                 -> simulation clock
    server_status[i]      -> 0 = idle, 1 = busy, for server i = 0,1
    num_busy_servers      -> sum(server_status)              -> represents B(t)
    num_in_queue          -> number of customers waiting     -> represents Q(t)

Time-weighted (area) accumulators:
    area_Q  -> integral of Q(t) dt                (time-average number in queue)
    area_B  -> integral of num_busy_servers(t) dt (used for server utilization)

Customer-average accumulators:
    total_delay          -> sum of waiting time in queue for every customer
    num_customers_served -> count of customers who have completed service
"""

import math
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. INPUT DATA: inter-arrival times and service times for each customer
#    (same arrays as the single-server version, unchanged)
# ---------------------------------------------------------------------
inter_arrival_times = [2, 3, 1, 4, 2, 5, 1, 3]   # time between successive arrivals
service_times       = [4, 2, 3, 1, 2, 3, 2, 4]   # service time required by each customer

N = min(len(inter_arrival_times), len(service_times))  # number of customers to simulate
NUM_SERVERS = 2

INFINITY = math.inf

# ---------------------------------------------------------------------
# 2. INITIALIZATION
# ---------------------------------------------------------------------
clock = 0.0
time_last_event = 0.0

server_status = [0] * NUM_SERVERS          # 0 = idle, 1 = busy, per server
next_departure_time = [INFINITY] * NUM_SERVERS   # scheduled departure time per server

num_in_queue = 0            # -> Q(t)
queue_arrival_times = []    # arrival times of customers currently waiting (FIFO)

area_Q = 0.0   # cumulative area under Q(t) curve
area_B = 0.0   # cumulative area under num_busy_servers(t) curve

# Lists to record state transitions over time for plotting Q(t) and B(t)
time_history = [0.0]
q_history = [0]
b_history = [0]
b0_history = [0]
b1_history = [0]

total_delay = 0.0
num_customers_served = 0
num_customers_delayed = 0

arrival_index = 0          # index into inter_arrival_times / who arrives next
service_index = 0          # index into service_times / whose service time is next used

next_arrival_time = inter_arrival_times[0]   # first arrival

print(f"{'Time':>7} | {'Event':<14} | {'Q(t)':>4} | {'Busy':>4} | Details")
print("-" * 80)


def num_busy_servers():
    return sum(server_status)


def find_idle_server():
    """Return index of an idle server, or None if all busy."""
    for i in range(NUM_SERVERS):
        if server_status[i] == 0:
            return i
    return None


# ---------------------------------------------------------------------
# 3. MAIN EVENT LOOP  (Next-Event Time Advance)
#    Stop when all customers have arrived AND the system is empty
#    (no one waiting, all servers idle).
# ---------------------------------------------------------------------
while arrival_index < N or num_busy_servers() > 0 or num_in_queue > 0:

    # ---- determine next event: soonest of arrival / either departure ----
    next_departure_time_overall = min(next_departure_time)
    next_event_time = min(next_arrival_time, next_departure_time_overall)

    # ---- advance time-weighted statistics up to next_event_time ----
    elapsed = next_event_time - time_last_event
    area_Q += num_in_queue * elapsed
    area_B += num_busy_servers() * elapsed
    time_last_event = next_event_time

    # ---- advance the clock ----
    clock = next_event_time

    # =========================== ARRIVAL EVENT ===========================
    if next_arrival_time <= next_departure_time_overall:

        idle_server = find_idle_server()

        if idle_server is None:
            # both servers busy -> customer joins the queue
            num_in_queue += 1
            queue_arrival_times.append(clock)
            print(f"{clock:7.2f} | {'ARRIVAL':<14} | {num_in_queue:4d} | {num_busy_servers():4d} | "
                  f"Customer {arrival_index + 1} arrives, both servers busy, joins queue")
        else:
            # an idle server takes the customer immediately, no delay
            server_status[idle_server] = 1
            num_customers_served += 1
            svc = service_times[service_index]
            service_index += 1
            next_departure_time[idle_server] = clock + svc
            print(f"{clock:7.2f} | {'ARRIVAL':<14} | {num_in_queue:4d} | {num_busy_servers():4d} | "
                  f"Customer {arrival_index + 1} arrives, starts service at server {idle_server} "
                  f"(delay=0.00, service time={svc})")

        arrival_index += 1
        if arrival_index < N:
            next_arrival_time = clock + inter_arrival_times[arrival_index]
        else:
            next_arrival_time = INFINITY   # no more arrivals scheduled

    # ========================== DEPARTURE EVENT ===========================
    else:
        # identify which server just finished
        server_id = next_departure_time.index(next_departure_time_overall)

        if num_in_queue == 0:
            # no one waiting -> this server becomes idle
            server_status[server_id] = 0
            next_departure_time[server_id] = INFINITY
            print(f"{clock:7.2f} | {'DEPARTURE':<14} | {num_in_queue:4d} | {num_busy_servers():4d} | "
                  f"Customer departs server {server_id}, queue empty, server {server_id} goes idle")
        else:
            # pull next customer from the queue into this now-free server
            num_in_queue -= 1
            cust_arrival = queue_arrival_times.pop(0)
            delay = clock - cust_arrival
            total_delay += delay
            num_customers_delayed += 1
            num_customers_served += 1

            svc = service_times[service_index]
            service_index += 1
            next_departure_time[server_id] = clock + svc

            print(f"{clock:7.2f} | {'DEPARTURE':<14} | {num_in_queue:4d} | {num_busy_servers():4d} | "
                  f"Customer departs server {server_id}, next customer starts service there "
                  f"(delay={delay:.2f}, service time={svc})")

    # Record state transition for plotting
    time_history.append(clock)
    q_history.append(num_in_queue)
    b_history.append(num_busy_servers())
    b0_history.append(server_status[0])
    b1_history.append(server_status[1])

# ---------------------------------------------------------------------
# 4. FINAL REPORT
# ---------------------------------------------------------------------
print("-" * 80)
print(f"Simulation ended at time {clock:.2f}")
print(f"Total customers processed        : {num_customers_served}")

avg_num_in_queue = area_Q / clock
server_utilization = area_B / (clock * NUM_SERVERS)   # fraction of total server-time busy
avg_delay = total_delay / N   # average delay over ALL customers (0 counted for those not delayed)

print(f"Time-average number in queue Q̄  : {avg_num_in_queue:.4f}")
print(f"Server utilization (rho)         : {server_utilization:.4f}")
print(f"Average delay in queue per cust. : {avg_delay:.4f}")

# ---------------------------------------------------------------------
# 5. PLOTTING Q(t) and B(t)
# ---------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Q(t) vs Time
ax1.step(time_history, q_history, where='post', color='royalblue', linewidth=2, label='Q(t)')
ax1.fill_between(time_history, q_history, step='post', alpha=0.2, color='royalblue')
ax1.set_ylabel('Number in Queue Q(t)', fontsize=11)
ax1.set_title('Double-Server Queue: Q(t) and B(t) vs Time', fontsize=13, fontweight='bold')
ax1.set_yticks(range(max(q_history) + 2))
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper right')

# Total B(t) vs Time
ax2.step(time_history, b_history, where='post', color='darkorange', linewidth=2, label='Total Busy Servers B(t)')
ax2.fill_between(time_history, b_history, step='post', alpha=0.2, color='darkorange')
ax2.set_ylabel('Total Busy Servers', fontsize=11)
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(['0', '1', '2'])
ax2.set_ylim(-0.1, 2.3)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc='upper right')

# Individual Server Status vs Time
ax3.step(time_history, b0_history, where='post', color='forestgreen', linewidth=2, label='Server 0 Status')
ax3.step(time_history, b1_history, where='post', color='mediumpurple', linewidth=2, linestyle='--', label='Server 1 Status')
ax3.set_xlabel('Simulation Time (t)', fontsize=11)
ax3.set_ylabel('Server Status', fontsize=11)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['0 (Idle)', '1 (Busy)'])
ax3.set_ylim(-0.1, 1.2)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.show()