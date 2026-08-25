# Flavor 1 — given the raw per-interval states, and you build the areas yourself
delays = [0.0, 0.8, 1.0, 0.0, 0.9, 3.0]
event_times = [0.0, 0.4, 1.6, 2.1, 2.4, 3.1, 3.3, 3.8, 4.0, 4.9, 5.6, 5.8, 7.2, 8.6]
Q_after = [0, 0, 1, 2, 1, 0, 0, 0, 1, 0, 1, 2, 3]   # queue length after each event
B_after = [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]   # server busy after each event

area_Q = sum(Q_after[i] * (event_times[i+1] - event_times[i]) for i in range(len(Q_after)))
area_B = sum(B_after[i] * (event_times[i+1] - event_times[i]) for i in range(len(B_after)))

avg_delay   = sum(delays) / len(delays)
avg_queue   = area_Q / event_times[-1]
utilization = area_B / event_times[-1]

# Flavor 2 — given the totals already (simplest possible exam question):
delays = [0.0, 0.8, 1.0, 0.0, 0.9, 3.0]
area_Q = 9.9
area_B = 7.7
T      = 8.6

avg_delay   = sum(delays) / len(delays)   # 0.95
avg_queue   = area_Q / T                  # 1.15
utilization = area_B / T                  # 0.90