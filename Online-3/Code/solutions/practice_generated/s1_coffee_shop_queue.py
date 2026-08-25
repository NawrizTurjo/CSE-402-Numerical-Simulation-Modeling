"""
Practice S1: Coffee Shop Queue (see ../../../Practice/PRACTICE_QUESTIONS.md).
Every task is just a different keyword argument to the shared
simulate_ssq engine -- see simulation/single_server_queue.py.

Run standalone:
    python -m solutions.practice_generated.s1_coffee_shop_queue
"""

from simulation.single_server_queue import simulate_ssq

INTERARRIVAL = [1.5, 0.8, 2.1, 0.3, 1.2, 0.9, 1.7, 0.4, 2.3, 0.6]
SERVICE = [2.5, 1.1, 0.9, 1.8, 0.7, 2.0, 1.3, 0.85]


def task1_single_server():
    r = simulate_ssq(INTERARRIVAL, SERVICE, num_delays=8)
    print("TASK 1: 1 barista, stop after 8 delays")
    print(f"  avg_delay={r['avg_delay']:.4f}  avg_queue={r['avg_queue']:.4f}  "
          f"utilization={r['utilization']:.4f}  T={r['T']:.2f}")
    print(f"  delays: {[round(d, 2) for d in r['delays']]}")
    print()


def task2_two_servers():
    r = simulate_ssq(INTERARRIVAL, SERVICE, num_delays=8, num_servers=2)
    print("TASK 2: 2 baristas, stop after 8 delays")
    print(f"  avg_delay={r['avg_delay']:.4f}  avg_queue={r['avg_queue']:.4f}  "
          f"utilization={r['utilization']:.4f}  T={r['T']:.2f}")
    print("  Every customer is served immediately (avg_delay=0) -- two servers")
    print("  comfortably absorb this arrival pattern, at the cost of ~47% idle")
    print("  server-time instead of ~13% (utilization roughly halves).")
    print()


def task3_balking():
    r = simulate_ssq(INTERARRIVAL, SERVICE, num_delays=8, capacity=2)
    print("TASK 3: 1 barista, waiting room capacity=2, stop after 8 delays")
    print(f"  avg_delay={r['avg_delay']:.4f}  avg_queue={r['avg_queue']:.4f}  "
          f"utilization={r['utilization']:.4f}  customers_turned_away={r['num_lost']}")
    print()


def task4_trace():
    r = simulate_ssq(INTERARRIVAL, SERVICE, num_delays=8, trace=True)
    print("TASK 4: trace of the first 6 events (1 barista)")
    for row in r["trace"][:6]:
        print(f"  t={row['clock']:.2f}  {row['event']:<42}  Q={row['queue_len']}  busy={row['busy_servers']}")
    print()


if __name__ == "__main__":
    task1_single_server()
    task2_two_servers()
    task3_balking()
    task4_trace()
