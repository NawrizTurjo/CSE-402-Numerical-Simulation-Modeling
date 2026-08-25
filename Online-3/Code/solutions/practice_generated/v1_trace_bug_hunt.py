"""
Practice V1: Verification by Trace -- Bug Hunt (see ../../../Practice/PRACTICE_QUESTIONS.md).

`buggy_simulate` below has ONE deliberate bug: it updates server/queue
state BEFORE accumulating the area-under-curve statistics, instead of
after (see simulation/single_server_queue.py's module docstring for why
the order matters). `simulate_ssq` (the library's correct engine) is the
"known-good" reference the slide's "verification by trace" technique
compares against.

Run standalone:
    python -m solutions.practice_generated.v1_trace_bug_hunt
"""

import heapq

from simulation.single_server_queue import simulate_ssq

INTERARRIVAL = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
SERVICE = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]


def buggy_simulate(interarrival, service, n_delays=6):
    """Same SSQ logic as simulate_ssq, but with the area-accumulation BUG."""
    arrival_times, t = [], 0.0
    for a in interarrival:
        t += a
        arrival_times.append(t)

    events = [(at, i, "A", i + 1) for i, at in enumerate(arrival_times)]
    heapq.heapify(events)

    last_event_time = 0.0
    busy = False
    queue = []
    area_Q = area_B = 0.0
    delays = []
    svc_iter = iter(service)
    seq = len(arrival_times)

    while events:
        clock, _, etype, cust = heapq.heappop(events)
        old_last_event_time = last_event_time  # keep the OLD boundary for area math below

        # --- BUG: state is updated FIRST ---
        if etype == "A":
            if not busy:
                busy = True
                delays.append(0.0)
                st = next(svc_iter, None)
                if st is not None:
                    heapq.heappush(events, (clock + st, seq, "D", cust))
                    seq += 1
            else:
                queue.append((cust, clock))
        else:
            busy = False
            if queue:
                next_cust, arrival_time = queue.pop(0)
                delays.append(clock - arrival_time)
                busy = True
                st = next(svc_iter, None)
                if st is not None:
                    heapq.heappush(events, (clock + st, seq, "D", next_cust))
                    seq += 1

        # --- then the area is accumulated using the ALREADY-UPDATED (wrong) state ---
        dt = clock - old_last_event_time
        area_Q += len(queue) * dt
        area_B += (1.0 if busy else 0.0) * dt
        last_event_time = clock

        if len(delays) == n_delays:
            break

    T = last_event_time
    return sum(delays) / len(delays), area_Q / T, area_B / T, T


def fixed_simulate(interarrival, service, n_delays=6):
    """
    The fix: accumulate area BEFORE updating state (swap the two blocks
    from buggy_simulate). This is exactly what simulate_ssq already does
    -- included here only so the diff against the buggy version is obvious.
    """
    arrival_times, t = [], 0.0
    for a in interarrival:
        t += a
        arrival_times.append(t)

    events = [(at, i, "A", i + 1) for i, at in enumerate(arrival_times)]
    heapq.heapify(events)

    last_event_time = 0.0
    busy = False
    queue = []
    area_Q = area_B = 0.0
    delays = []
    svc_iter = iter(service)
    seq = len(arrival_times)

    while events:
        clock, _, etype, cust = heapq.heappop(events)

        # --- FIX: accumulate area using the OLD state first ---
        dt = clock - last_event_time
        area_Q += len(queue) * dt
        area_B += (1.0 if busy else 0.0) * dt
        last_event_time = clock

        if etype == "A":
            if not busy:
                busy = True
                delays.append(0.0)
                st = next(svc_iter, None)
                if st is not None:
                    heapq.heappush(events, (clock + st, seq, "D", cust))
                    seq += 1
            else:
                queue.append((cust, clock))
        else:
            busy = False
            if queue:
                next_cust, arrival_time = queue.pop(0)
                delays.append(clock - arrival_time)
                busy = True
                st = next(svc_iter, None)
                if st is not None:
                    heapq.heappush(events, (clock + st, seq, "D", next_cust))
                    seq += 1

        if len(delays) == n_delays:
            break

    T = last_event_time
    return sum(delays) / len(delays), area_Q / T, area_B / T, T


def task1_compare():
    d, q, u, T = buggy_simulate(INTERARRIVAL, SERVICE)
    print("TASK 1: buggy version vs the known-correct expected output")
    print(f"  BUGGY    -> avg_delay={d:.4f}  avg_queue={q:.4f}  utilization={u:.4f}  T={T:.1f}")
    print(f"  EXPECTED -> avg_delay=0.9500  avg_queue=1.1512  utilization=0.8953  T=8.6")
    print("  avg_delay matches, but avg_queue and utilization do not.")
    print()


def task2_explain():
    print("TASK 2: why the bug produces wrong numbers")
    print(
        "  area_Q += len(queue)*dt is meant to add the rectangle 'queue length\n"
        "  held DURING the interval [last_event_time, clock]' -- i.e. the state\n"
        "  that was true BEFORE this event fired, not after. buggy_simulate\n"
        "  updates queue/busy for the event that just happened, THEN accumulates\n"
        "  area using that already-updated state -- attributing the new state to\n"
        "  the interval that came before it, which inflates both areas."
    )
    print()


def task3_verify_fix():
    d, q, u, T = fixed_simulate(INTERARRIVAL, SERVICE)
    print("TASK 3: fixed version, re-verified against the library")
    print(f"  FIXED   -> avg_delay={d:.4f}  avg_queue={q:.4f}  utilization={u:.4f}  T={T:.1f}")

    r = simulate_ssq(INTERARRIVAL, SERVICE, num_delays=6)
    print(f"  LIBRARY -> avg_delay={r['avg_delay']:.4f}  avg_queue={r['avg_queue']:.4f}  "
          f"utilization={r['utilization']:.4f}  T={r['T']:.1f}")
    print()


def task4_reflect():
    print("TASK 4: which V&V technique, and why avg_delay survived the bug")
    print(
        "  This is VERIFICATION BY TRACE: comparing a program's output against\n"
        "  independently known-correct values to catch implementation bugs --\n"
        "  as opposed to VALIDATION (does the model represent the real system\n"
        "  correctly in the first place?), a different question not exercised\n"
        "  here. avg_delay survives because the delay calculation\n"
        "  (clock - arrival_time when service starts) never touches area_Q or\n"
        "  area_B -- it's a separate accumulator, so a bug isolated to the area\n"
        "  bookkeeping only breaks the two measures actually built from it."
    )


if __name__ == "__main__":
    task1_compare()
    task2_explain()
    task3_verify_fix()
    task4_reflect()
