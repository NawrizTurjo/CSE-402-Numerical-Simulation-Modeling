"""
ONE discrete-event simulation engine that covers every queueing variant
seen across the practice material:

    - stop after N customer DELAYS have been observed   (num_delays=N)
    - stop at a fixed simulation TIME                     (t_max=T)
    - run until the system fully DRAINS (all arrivals processed,
      all servers idle, queue empty)                       (neither given)
    - c parallel identical servers instead of 1             (num_servers=c)
    - a finite waiting room / balking (customers leave if
      the queue is already full)                            (capacity=k)
    - an optional printed trace table                        (trace=True)

Rather than five separate near-duplicate scripts (one per variant, which
is what you'll find if you go looking at old exam solutions), this is
the SAME next-event time-advance loop every time; only which stopping
condition fires, and how many servers there are, changes. Learn this one
function and every variant is just "which keyword argument do I pass".

Next-event time-advance mechanics (the general idea, independent of any
variant):
    - Precompute arrival times = cumulative sum of interarrival times.
    - Keep a min-heap ("event list") of (time, ..., type, customer) events.
    - Pop the earliest event, ADVANCE the clock to it.
    - Before changing any state, accumulate area-under-curve statistics
      using the OLD state over the interval [last_event_time, clock] --
      this is the single most common bug source: if you update the state
      first, the rectangle you integrate has the WRONG height.
    - Then update state (server busy/idle, queue) and schedule whatever
      new event(s) this one triggers.

Performance measures, from the accumulated areas:
    avg_delay    = total customer delay / number of customers delayed
    avg_queue    = area_Q / T         (time-average number waiting)
    utilization  = area_B / (num_servers * T)   (fraction of server-time busy)

Run standalone:
    python -m simulation.single_server_queue
"""


import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import heapq
import itertools
import math


def simulate_ssq(interarrival_times, service_times, *, num_delays=None, t_max=None,
                  num_servers=1, capacity=None, trace=False, plot=False):
    """
    Parameters
    ----------
    interarrival_times : list of gaps between successive arrivals
                          (arrival_k's clock time = sum of the first k of these).
    service_times      : list of service durations, consumed in FIFO order
                          as customers begin service (server assignment order,
                          not arrival order, for multi-server -- but with
                          num_servers=1 the two coincide).
    num_delays  : stop as soon as this many customers have BEGUN service
                  (i.e. this many delays have been observed). None = no limit.
    t_max       : stop at this simulation time (partial last interval is
                  still counted -- the "tail rectangle"). None = no limit.
                  If both num_delays and t_max are given, whichever
                  triggers first stops the simulation.
    num_servers : number of identical parallel servers (c). Default 1.
    capacity    : max customers allowed to WAIT in queue. A customer who
                  arrives when the queue is already at capacity balks
                  (leaves immediately, counted in `num_lost`). None = unlimited.
    trace       : if True, also return a list of per-event trace rows.

    Returns a dict: avg_delay, avg_queue, utilization, T (sim end time),
    delays (list), num_lost, num_customers_delayed, and trace (if requested).
    """
    arrival_times = []
    t = 0.0
    for gap in interarrival_times:
        t += gap
        arrival_times.append(t)

    seq = itertools.count()  # heap tie-breaker so events at equal time never compare types/ids directly
    events = []
    for i, at in enumerate(arrival_times):
        if t_max is not None and at > t_max:
            break
        heapq.heappush(events, (at, next(seq), "A", i + 1))

    clock = last_event_time = 0.0
    busy_servers = 0
    queue = []  # FIFO list of (customer_id, arrival_time) waiting for a server
    area_Q = area_B = 0.0
    delays = []
    num_lost = 0
    svc_iter = iter(service_times)
    trace_rows = []

    stop_reason = "drained"  # overwritten below if we stop early instead of exhausting events

    while events:
        if t_max is not None and events[0][0] > t_max:
            stop_reason = "t_max"
            break

        clock, _, etype, cust = heapq.heappop(events)

        # Accumulate areas using the OLD state, over the interval that just ended.
        dt = clock - last_event_time
        area_Q += len(queue) * dt
        area_B += busy_servers * dt
        last_event_time = clock

        if etype == "A":
            if busy_servers < num_servers:
                busy_servers += 1
                delays.append(0.0)  # server was free -> zero wait
                service_time = next(svc_iter, None)
                if service_time is not None:
                    heapq.heappush(events, (clock + service_time, next(seq), "D", cust))
                event_desc = f"Arrival C{cust} (starts service, D=0)"
            elif capacity is not None and len(queue) >= capacity:
                num_lost += 1
                event_desc = f"Arrival C{cust} (BALKS - queue full)"
            else:
                queue.append((cust, clock))
                event_desc = f"Arrival C{cust} (joins queue)"
        else:  # Departure
            busy_servers -= 1
            if queue:
                next_cust, arrival_time = queue.pop(0)
                delay = clock - arrival_time
                delays.append(delay)
                busy_servers += 1
                service_time = next(svc_iter, None)
                if service_time is not None:
                    heapq.heappush(events, (clock + service_time, next(seq), "D", next_cust))
                event_desc = f"Departure C{cust} (C{next_cust} starts service, D={delay:.2f})"
            else:
                event_desc = f"Departure C{cust} (server goes idle)"

        if trace:
            trace_rows.append({
                "clock": clock, "event": event_desc,
                "queue_len": len(queue), "busy_servers": busy_servers,
                "area_Q": area_Q, "area_B": area_B,
            })

        if num_delays is not None and len(delays) == num_delays:
            stop_reason = "num_delays"
            break

    # Tail rectangle: only needed when we stopped at a fixed t_max with state
    # still "in progress" -- num_delays/drained stops already reflect the true end time.
    if stop_reason == "t_max":
        dt = t_max - last_event_time
        area_Q += len(queue) * dt
        area_B += busy_servers * dt
        T = t_max
    else:
        T = last_event_time

    result = {
        "avg_delay": sum(delays) / len(delays) if delays else 0.0,
        "avg_queue": area_Q / T if T else 0.0,
        "utilization": area_B / (num_servers * T) if T else 0.0,
        "T": T,
        "delays": delays,
        "num_lost": num_lost,
        "num_customers_delayed": len(delays),
        "stop_reason": stop_reason,
    }
    if trace or plot:
        result["trace"] = trace_rows
    if plot and trace_rows:
        plot_ssq_trace(trace_rows)
    return result


def plot_ssq_trace(trace_rows, title="Queue Simulation: Q(t) and B(t) vs Time", show=True, save_path=None):
    """
    Plot step curves of Q(t) and B(t) over simulation time (Law & Kelton style).

    Parameters
    ----------
    trace_rows : list of dict
        Trace rows from simulate_ssq(..., trace=True)['trace'].
    title : str, default='Queue Simulation: Q(t) and B(t) vs Time'
        Figure title.
    show : bool, default=True
        Whether to call plt.show().
    save_path : str, optional
        If provided, save the figure to this file path.
    """
    import matplotlib.pyplot as plt

    times = [0.0] + [r["clock"] for r in trace_rows]
    q_vals = [0] + [r["queue_len"] for r in trace_rows]
    b_vals = [0] + [r["busy_servers"] for r in trace_rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Q(t) vs Time
    ax1.step(times, q_vals, where="post", color="royalblue", linewidth=2, label="Q(t)")
    ax1.fill_between(times, q_vals, step="post", alpha=0.2, color="royalblue")
    ax1.set_ylabel("Number in Queue Q(t)", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.set_yticks(range(max(q_vals) + 2))
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(loc="upper right")

    # B(t) vs Time
    ax2.step(times, b_vals, where="post", color="darkorange", linewidth=2, label="B(t)")
    ax2.fill_between(times, b_vals, step="post", alpha=0.2, color="darkorange")
    ax2.set_xlabel("Simulation Time (t)", fontsize=11)
    ax2.set_ylabel("Server Status B(t)", fontsize=11)
    max_b = max(b_vals) if b_vals else 1
    ax2.set_yticks(range(max_b + 2))
    ax2.set_ylim(-0.1, max_b + 0.3)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def compute_performance_from_trace(delays, event_times, queue_len_after, busy_after, num_servers=1):
    """
    Use this INSTEAD of simulate_ssq when the exam question hands you a
    finished trace table directly (event times + queue length/server
    status after each event) and just asks for the summary statistics --
    no simulation loop needed, just the area-under-curve arithmetic.

    delays          : list of each customer's observed delay.
    event_times     : [t0=0.0, t1, ..., tN=T]  (N+1 timestamps, start to end).
    queue_len_after : queue length held during [event_times[i], event_times[i+1]],
                       i.e. the value immediately AFTER event i fires (N entries).
    busy_after      : number of busy servers over the same intervals (N entries).
    """
    T = event_times[-1]
    area_Q = sum(queue_len_after[i] * (event_times[i + 1] - event_times[i])
                 for i in range(len(queue_len_after)))
    area_B = sum(busy_after[i] * (event_times[i + 1] - event_times[i])
                 for i in range(len(busy_after)))

    return {
        "avg_delay": sum(delays) / len(delays),
        "avg_queue": area_Q / T,
        "utilization": area_B / (num_servers * T),
        "T": T,
    }


if __name__ == "__main__":
    # Slide's worked example: single server, stop after 6 delays.
    interarrival = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
    service = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]

    print("=== Variant: stop after N=6 delays ===")
    r = simulate_ssq(interarrival, service, num_delays=6)
    print(f"avg_delay={r['avg_delay']:.4f}  avg_queue={r['avg_queue']:.4f}  "
          f"utilization={r['utilization']:.4f}  T={r['T']:.1f}")
    # Expected: avg_delay=0.9500  avg_queue=1.1512  utilization=0.8953  T=8.6

    print("\n=== Variant: stop at fixed T_max=8.6 ===")
    r = simulate_ssq(interarrival, service, t_max=8.6)
    print(f"avg_delay={r['avg_delay']:.4f}  avg_queue={r['avg_queue']:.4f}  "
          f"utilization={r['utilization']:.4f}  T={r['T']:.1f}")

    print("\n=== Variant: run to completion (drain), trace on ===")
    # Drain needs a service time available for every customer who will ever
    # start service, so use matched-length arrays here (the 9-vs-6 arrays
    # above are only long enough for the first 6 delays, as in the slide's
    # worked example).
    interarrival_drain = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6]
    service_drain = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]
    r = simulate_ssq(interarrival_drain, service_drain, trace=True)
    for row in r["trace"]:
        print(f"  t={row['clock']:5.2f}  {row['event']:<38}  Q={row['queue_len']}  busy={row['busy_servers']}")
    print(f"Final: avg_delay={r['avg_delay']:.4f}  avg_queue={r['avg_queue']:.4f}  "
          f"utilization={r['utilization']:.4f}  T={r['T']:.1f}  stop_reason={r['stop_reason']}")

    print("\n=== Variant: 2 servers, stop after N=6 delays ===")
    r = simulate_ssq(interarrival, service, num_delays=6, num_servers=2)
    print(f"avg_delay={r['avg_delay']:.4f}  avg_queue={r['avg_queue']:.4f}  utilization={r['utilization']:.4f}")

    print("\n=== From a given trace table directly (no simulation) ===")
    delays = [0.0, 0.8, 1.0, 0.0, 0.9, 3.0]
    event_times = [0.0, 0.4, 1.6, 2.1, 2.4, 3.1, 3.3, 3.8, 4.0, 4.9, 5.6, 5.8, 7.2, 8.6]
    Q_after = [0, 0, 1, 2, 1, 0, 0, 0, 1, 0, 1, 2, 3]
    B_after = [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    r = compute_performance_from_trace(delays, event_times, Q_after, B_after)
    print(f"avg_delay={r['avg_delay']:.4f}  avg_queue={r['avg_queue']:.4f}  utilization={r['utilization']:.4f}")
