"""
Visual diagnostics for a discrete-event queue simulation: the step
functions Q(t) (number in queue) and B(t) (busy servers) over simulated
time. Res/Src-29/dsq_des.py built these by hand, appending to parallel
time_history/q_history/b_history lists inside the simulation loop
itself; here they're reconstructed straight from the `trace` that
simulation/single_server_queue.py's simulate_ssq(..., trace=True) already
returns, so there's no second bookkeeping pass to keep in sync with the
engine.

Run standalone:
    python -m viz.des_plots
"""

import matplotlib.pyplot as plt


def plot_queue_and_server(trace, num_servers=1, title="SSQ Simulation", show=True, save_path=None):
    """
    trace : the list of per-event dicts from
            simulate_ssq(..., trace=True)["trace"] -- each row needs
            'clock', 'queue_len', 'busy_servers'.
    num_servers : only used to label/scale the utilization axis.

    Prepends the t=0 idle/empty starting state, then plots both series
    with matplotlib's where='post' step mode: the value recorded at
    trace row i is the state that becomes true AT that row's clock time
    and holds until the next event -- exactly what where='post' draws.
    """
    times = [0.0] + [row["clock"] for row in trace]
    queue_lens = [0] + [row["queue_len"] for row in trace]
    busy = [0] + [row["busy_servers"] for row in trace]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.step(times, queue_lens, where="post", color="royalblue", linewidth=2)
    ax1.fill_between(times, queue_lens, step="post", alpha=0.2, color="royalblue")
    ax1.set_ylabel("Q(t) -- number in queue")
    ax1.set_title(title)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.step(times, busy, where="post", color="darkorange", linewidth=2)
    ax2.fill_between(times, busy, step="post", alpha=0.2, color="darkorange")
    ax2.set_xlabel("Simulation time")
    ax2.set_ylabel(f"B(t) -- busy servers (of {num_servers})")
    ax2.set_yticks(range(num_servers + 1))
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    from simulation.single_server_queue import simulate_ssq

    interarrival = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
    service = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]
    result = simulate_ssq(interarrival, service, num_delays=6, trace=True)

    plot_queue_and_server(
        result["trace"], num_servers=1,
        title="Slide's worked example (6 delays)",
        show=False, save_path="des_plot_demo.png",
    )
    print("Saved des_plot_demo.png")
    print(f"avg_delay={result['avg_delay']:.4f}  avg_queue={result['avg_queue']:.4f}  "
          f"utilization={result['utilization']:.4f}")
    print("(pass show=True, save_path=None for the normal interactive exam usage)")
