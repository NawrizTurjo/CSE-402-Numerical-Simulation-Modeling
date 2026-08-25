# CSE401: Discrete-Event Simulation & Single-Server Queue Simulation
Source: `5_simu_des_ssqs.pdf` (154 pages / 89 numbered slides), Nafis Tahmid, CSE, BUET.
Topics: Intro to Simulation & Modeling; Discrete Event Simulation Models; Steps in a Simulation Study; Model Validation and Verification; Single-Server Queue (SSQ) hand simulation.

---

## 1. Introduction to Simulation

**Why not experiment on the real system?** Real experiments can be expensive, dangerous, or embarrassing (e.g., firing ER doctors to see effect on wait times; building a US$300M port berth to test a design). Simulation lets you make mistakes cheaply and safely.

**Definition — Simulation:** *The imitation of the operation of a real-world process or system over time.*
- Imitation = we create a fake version of something real
- Real-world process/system = something that exists or could exist
- Over time = we watch it evolve, step by step

**In plain English:** We generate an artificial history of a system, observe it, and draw conclusions about how the real system would behave.

**Artificial History Idea (pipeline):**
```
Real System --simplify--> Model --run--> Simulated Data --analyze--> Insights!
```
We build a simplified Model of the real system, run it on a computer, collect data as if observing the real system, then use that data to estimate real-system performance.

### Simulation vs. Analytical Solutions
- If a system is simple enough, write equations and solve directly → **analytical solution** (e.g., Distance = Rate × Time).
- Complex systems (e.g., avg wait time at a 3-teller bank with random arrivals/services) → math becomes intractable → use **simulation**.

| System complexity | Use |
|---|---|
| Simple | Math (analytical solution) |
| Complex | Simulation |

### When IS Simulation the Right Tool?
1. System too complex to solve mathematically
2. Want to experiment without disrupting the real system
3. System doesn't exist yet (still in design)
4. Want to study system under extreme conditions ("what if 1000 patients arrive at once?")
5. Want to compress/expand time (simulate a year in minutes, or slow down a nanosecond process)
6. Want to train people without real-world risk (flight simulators, military exercises)
7. Want to test new designs before committing resources

### When Simulation is NOT Appropriate
- **Rule 1:** Don't simulate if common sense suffices (e.g., arrivals 100/hr, each server handles 12/hr → need ≥ 9 servers; just divide, no sim needed).
- **Rule 2:** Don't simulate if the problem can be solved analytically (clean formula exists → faster, exact; simulation only gives estimates).
- **Rule 3:** Don't simulate if direct experiments are cheaper (e.g., testing a new ordering station by just trying it in person instead of building a model).
- Other reasons not to simulate: cost of study exceeds potential savings; not enough time/budget; no data available (not even estimates); system behavior too complex to even define.

### Where Simulation is Used
Manufacturing (assembly lines, wafer fab, bottleneck analysis) · Military (weapons evaluation, training, logistics) · Healthcare (ER design, OR scheduling, disease spread) · Business (call centers, supply chains, inventory) · Transportation (runway mgmt, traffic flow, ports) · Networks/construction/predator-prey ecosystems, etc.

---

## 2. Systems, Models, and Classifications

### System
**Definition:** A system is a collection of entities (people, machines, etc.) that act and interact together toward the accomplishment of some logical end.
Examples: bank (tellers+customers+queues+rules), factory (machines+workers+parts+conveyors), hospital, computer network.
**Key point:** what counts as "the system" depends on the purpose of the study.

### System Boundary & Environment
- The **system boundary** depends on study purpose. E.g., studying teller wait times → system = tellers + customers in line (loan officer, safe-deposit boxes excluded). Studying whole bank operation → system = tellers + loan officers + safe-deposit + ATMs + ...
- **System environment:** everything outside the system boundary. The environment can affect the system, but the system doesn't control it (e.g., customer arrival pattern is part of the bank's environment).

### Components of a System
- **Entity:** an object of interest in the system
- **Attribute:** a property of an entity
- **Activity:** a time period of specified length
- **Event:** an instantaneous occurrence that may change the system state
- **State variable:** a variable describing the system at any time

Bank example: Entity=Customers, Attribute=Account balance, Activity=Making a deposit, Event=Customer arrives/departs, State variable=Number of busy tellers, number waiting.

**More examples table:**

| System | Entity | Attribute | Activity | Event | State Var. |
|---|---|---|---|---|---|
| Bank | Customers | Balance | Deposit | Arrival | # busy tellers |
| Rail | Riders | Origin | Traveling | Arr. at station | # waiting |
| Factory | Machines | Speed | Stamping | Breakdown | Status |
| Network | Messages | Length | Transmit | Arr. at destination | # waiting |
| Inventory | Warehouse | Capacity | Withdraw | Demand | Stock level |

Note: components can only be listed once the study's purpose is known — different purposes → different relevant components.

### State Variables
The **state** of a system = the collection of variables describing the system at a point in time (a "snapshot" sufficient to reconstruct exactly what's happening).
Bank example state: # busy tellers=2, # customers in queue=5, time of next arrival=10:23 AM.
**Important:** State variables change only when events occur. Between events, nothing changes (from the model's perspective).

### Discrete vs. Continuous Systems

| Discrete System | Continuous System |
|---|---|
| State variables change at discrete (separated) points in time | State variables change continuously over time |
| Example: a bank — # customers changes only when someone arrives/departs (step function graph, e.g. 1→2→3→2→1) | Example: water level behind a dam — rises/falls smoothly (smooth curve) |

Most real systems are a mix; classify by the **predominant** type of change.

### Model
**Definition:** A model is a representation of a system for the purpose of studying that system. A model is, by definition, a simplification — we model only the parts that matter for our question.
Analogy: A map of Dhaka is a model of Dhaka — doesn't show every brick/person/cat, but shows roads/landmarks/distances needed for navigation.

### How to Study a System (diagram, tree)
```
                         System
                /                    \
  Experiment with              Experiment with
   actual system                    a model
                              /                  \
                       Physical model      Mathematical model
                                            /              \
                                  Analytical solution    Simulation
```
Simulation is used when we have a mathematical model too complex to solve analytically.

### Types of Simulation Models (3 classification dimensions)
1. **Static vs. Dynamic**
   - Static: represents system at a single point in time (e.g., Monte Carlo)
   - Dynamic: represents how a system evolves over time (e.g., bank from 9 AM–4 PM)
2. **Deterministic vs. Stochastic**
   - Deterministic: no randomness; same inputs → same outputs
   - Stochastic: contains random variables; outputs are estimates
3. **Discrete vs. Continuous**
   - Discrete: state changes at separated points in time
   - Continuous: state changes continuously

**Our focus: Discrete-Event Simulation (DES)** = Dynamic + Stochastic + Discrete.
Why stochastic? The real world is random — customers don't arrive at exact intervals, service times vary, machines break down unpredictably. Ignoring randomness gives a dangerously wrong model.

**Deterministic vs Stochastic intuition:**
- Deterministic: dentist's office, every patient arrives exactly at scheduled time (unrealistic but predictable).
- Stochastic: bank, customers walk in randomly, service times vary — randomness creates queues/delays/interesting behavior.
- **Key consequence:** because inputs are random, outputs are random too. Running the same model twice with different random numbers → different results. Simulation outputs are *estimates*, not exact answers — need statistics to interpret them.

---

## 3. Discrete-Event Simulation (DES) — Mechanics

**Definition:** DES is the modeling of a system in which the state variables change only at a discrete set of points in time, called **events**. Between events, nothing changes.
Bank example: Customer arrives at 10:03 → state changes (queue+1). Service completes at 10:07 → state changes (queue−1). Between 10:03–10:07, queue length is constant.

### The Simulation Clock
- Every DES model has a variable called the **simulation clock** — records the current value of simulated time.
- Simulated time ≠ actual computer time. You could simulate 24 hrs of bank operation in 0.01s of computer time, or 1 nanosecond of a computer network in 10 minutes of computer time.

### How the Clock Moves Forward — Two Approaches
1. **Next-Event Time Advance** (the one used in this course / virtually all modern software)
   - Jump the clock from one event straight to the next
   - Skip over "boring" periods where nothing happens
2. **Fixed-Increment Time Advance**
   - Advance the clock by a fixed Δt each step; check if any events occurred during [t, t+Δt)
   - Wasteful — many steps where nothing happens

**Next-Event Time Advance — the idea (algorithm):**
1. Start the clock at 0
2. Determine when future events will occur
3. Jump the clock to the nearest upcoming event
4. Process that event (update state, schedule new events)
5. Repeat until done

Diagram (timeline): clock jumps `0 → e1 → e2 → e3 → e4 → e5`, each jump labeled "jump", with idle gaps between events skipped entirely (labeled "skip idle"). Contrast: fixed-increment walks small steps continuously ("checking... nothing... checking...") with only occasional steps landing on an actual "event!".

### Components of a DES Model (as programmed)
- **System state:** variables describing the system right now
- **Simulation clock:** current simulated time
- **Event list** (a.k.a. future event list): list of upcoming events and when they'll occur
- **Statistical counters:** variables accumulating performance data
- **Initialization routine:** sets everything up at time 0
- **Timing routine:** finds the next event and advances the clock
- **Event routines:** update state when an event occurs
- **Library routines:** generate random numbers
- **Report generator:** computes and prints results
- **Main program:** orchestrates everything

### The Event List
The event list (future event list) = list of events scheduled to happen in the future, with the time each will occur.

Example:
| Event Type | Scheduled Time |
|---|---|
| Customer 4 arrives | 10:15 |
| Customer 2 finishes service | 10:08 |
| Customer 5 arrives | 10:22 |

The timing routine scans this list and picks the **earliest** event (here: Customer 2 finishes at 10:08).

### Flow of Control in a DES (full flowchart, transcribed)

```
                              [ Start ]
                                  |
                                  v
  +----------------------+   +----------------------+   +----------------------+
  | Initialization routine|<-0-| Main program         |-i->| Timing routine       |
  | 1. Set sim clock = 0  |   | 0. Invoke init routine|   | 1. Determine next    |
  | 2. Init system state &|   | 1. Invoke timing rtn  |<-Repeatedly-  event type i    |
  |    statistical counters|  | 2. Invoke event rtn i |    i         | 2. Advance sim clock |
  | 3. Init event list    |   +----------------------+   +----------------------+
  +-----------+-----------+          ^      |
              |                      |      v
              v                      |  +----------------------+   +----------------------+
              +--------------------->|  | Event routine i       |<->| Library routines     |
                                     |  | 1. Update system state |   | Generate random      |
                                     |  | 2. Update stat. counters|  | variates             |
                                     |  | 3. Generate future events|  +----------------------+
                                     |  |    & add to event list |
                                     |  +-----------+------------+
                                     |              |
                                     |              v
                                     |     < Is simulation over? >---No---(back to Main program / Timing)
                                     |              |
                                     |             Yes
                                     |              v
                                     |  +----------------------+
                                     |  | Report generator      |
                                     |  | 1. Compute estimates   |
                                     |  | 2. Write report        |
                                     |  +-----------+------------+
                                     |              v
                                     |          [ Stop ]
```

**Flow of control — textual algorithm (equivalent, this is the version to memorize for coding):**
1. **Start:** Main program invokes the **initialization routine**
   - Set simulation clock = 0
   - Initialize system state and statistical counters
   - Schedule initial events in the event list
2. **Repeatedly** (main loop):
   1. Main program calls **timing routine** → determines next event type *i* and advances clock
   2. Main program calls **event routine i** → updates state, updates counters, schedules future events (may call **library routines** to generate random variates)
   3. Check: is the simulation over? → **No:** go back to step 2.1 of loop → **Yes:** invoke the **report generator** and stop

---

## 4. Single-Server Queueing System (SSQ)

### Introduction to Queueing
A queue = a line: you stand in it, wait, get served, leave. Queues form whenever demand (temporarily) exceeds supply. Understanding queues → design better systems (shorter waits, lower cost, happier people). Queues are everywhere: bank tellers, traffic signals, hospitals, CPU job queues, call centers, network routers.

### Components of a Queueing System (diagram)
```
 (Calling population) --Arrivals--> [ Waiting line (Queue): ●●●● ] --> [ Server ] --Departures-->
```
1. **Calling population:** where customers come from
2. **Arrival process:** how customers arrive (interarrival times)
3. **Service mechanism:** how many servers, how long service takes
4. **Queue discipline:** who gets served next (FIFO, priority, etc.)
5. **System capacity:** is there a limit on how many can wait

### Arrival Process
Customers arrive one at a time at random intervals. The time between two consecutive arrivals = **interarrival time**. Denote interarrival time between customer *i−1* and customer *i* as **Aᵢ**.

Diagram: on a timeline, customers C1..C5 arrive with gaps A2, A3, A4, A5 between them (unequal — bursts and long gaps both occur; this randomness is the essence of queueing).

**Arrival time of customer i** = cumulative sum of interarrival times:
- Arrival time of customer 1 = A1
- Arrival time of customer 2 = A1 + A2
- Arrival time of customer i = Σ_{j=1}^{i} Aⱼ

### What Happens When a Customer Arrives (flowchart)
```
Customer arrives
      |
      v
 < Server busy? > --No--> Begin service immediately
      |
     Yes
      |
      v
 Join the queue and wait
```
- **Server idle:** zero delay, straight to service.
- **Server busy:** customer joins end of queue, waits.

### What Happens When Service Completes (flowchart)
```
Service completes (customer departs)
      |
      v
 < Queue empty? > --Yes--> Server becomes idle
      |
      No
      |
      v
 Next customer enters service
```
- **Queue empty:** server becomes idle; set next departure time to ∞.
- **Queue not empty:** first person in line moves to server; generate their service time and schedule their departure.

### Our Setup: The Single-Server System
- One server (e.g., one barber, one ATM, one teller)
- Customers arrive one at a time, at random intervals
- Server idle → customer goes straight to service (delay = 0)
- Server busy → customer joins end of queue
- **Queue discipline: FIFO** (first in, first out)
- Start **empty and idle**: no customers, server idle

**Notation:**
- **Aᵢ** = interarrival time between customer i−1 and i
- **Sᵢ** = service time of customer i
- **Dᵢ** = delay in queue of customer i (waiting time before service begins)

### Delay vs. Total Time in System
- **Delay Dᵢ** = time customer i spends waiting in queue, NOT being served. If server idle on arrival, delay = 0.
- **Total time in system** = delay + service time = Dᵢ + Sᵢ.
- We focus on Dᵢ because it measures system *inefficiency*. Service time is inherent to the task; waiting time is waste.

### What the Computer Must Track
1. **Clock:** current simulated time
2. **Server status:** 0 (idle) or 1 (busy)
3. **Number in queue:** how many customers waiting
4. **Times of arrival:** list storing arrival time of each customer currently in queue (needed to compute delays later)
5. **Time of last event:** when the previous event occurred (needed for area calculations)
6. **Event list:** times of next arrival and next departure

**Statistical counters:**
- Total delay accumulated
- Number delayed (so far)
- Area under B(t) (so far)
- Area under Q(t) (so far)

---

## 5. Performance Measures (Formulas)

Three things to measure: (1) how long customers wait, (2) how crowded the queue is, (3) how busy the server is.

### Measure 1: Average Delay in Queue, d̂(n)
After n customers have completed their delays:
```
d̂(n) = (D1 + D2 + ... + Dn) / n  =  (1/n) * Σ_{i=1}^{n} Di
```
**Worked mini-example:** D1=0, D2=0.8, D3=1.0, D4=0, D5=0.9, D6=3.0
```
d̂(6) = (0+0.8+1.0+0+0.9+3.0)/6 = 5.7/6 = 0.95
```

### Measure 2: Time-Average Number in Queue, q̂(n)
Not a simple average — must be **time-weighted**. (Illustration: queue has 0 customers for 9 hrs, 10 customers for 1 hr. Simple average of observed values (0+10)/2 = 5 is WRONG — queue was empty 90% of the time; correct answer closer to (0·9+10·1)/10 = 1.0.)

Define **Q(t)** = number of customers in queue at time t (not counting the one in service). Q(t) is a **step function** — changes only at event times, constant between events.

```
q̂(n) = [ ∫_0^{T(n)} Q(t) dt ] / T(n)
```
where T(n) = time the simulation ends.

Since Q(t) is a step function, the integral is just a **sum of rectangle areas**:
```
∫_0^{T(n)} Q(t) dt = Σ_over_each_interval (queue length during interval) × (duration of interval)
```
Each rectangle: height = value of Q(t) during interval, width = time from one event to next.

**Worked mini-example (Q(t) step function graph):** Shaded area = ∫₀^T Q(t)dt = 9.9, T=8.6 → q̂(6) = 9.9/8.6 = 1.15 customers on average in queue.

### Measure 3: Server Utilization, û(n)
Define the busy function:
```
B(t) = 1  if server busy at time t
B(t) = 0  if server idle at time t
```
Server utilization = time-average of B(t):
```
û(n) = [ ∫_0^{T(n)} B(t) dt ] / T(n) = (total time server was busy) / T(n)
```
Since B(t) is always 0 or 1, the integral is just total busy time. Result ∈ [0,1]:
- û = 0: server idle entire time (no customers ever came)
- û = 1: server busy entire time (never idle)
- û = 0.9: server busy 90% of the time

**Why utilization matters:** near 100% → server is a bottleneck, queues grow; very low → excess/idle capacity; sweet spot usually 70–85%. **Rule of thumb:** if arrival rate > service rate, queue grows without bound ("explosive" queue).

**Worked mini-example (B(t) graph):** Busy intervals 2.9 and 4.8 (total busy = 7.7), T=8.6 → û(6) = 7.7/8.6 = 0.90 (90% busy).

### How We Actually Compute the Areas (algorithm — critical for coding)
We do NOT draw the graph and measure with a ruler — we **accumulate areas as the simulation runs, event by event**.

At each event, BEFORE updating the state, add a rectangle:
```
new_area += (current value of Q(t) or B(t)) * (current_time - time_of_last_event)
              ^ height (OLD/previous state value)      ^ width
```
**Order matters (must memorize for the exam):**
1. **First:** Update area accumulators using the OLD (pre-event) values of Q(t) and B(t)
2. **Then:** Update state variables (server status, queue length, arrival list)
3. **Then:** Update "time of last event" to current clock value
4. **Then:** Update event list (schedule new arrivals/departures)

**Common bugs:**
- Updating Q(t)/state before computing the area → wrong rectangle height
- Updating time-of-last-event before computing areas → wrong rectangle width
- Forgetting to set departure time to ∞ when queue empties → phantom departures

---

## 6. Worked Example: Hand-Simulating an SSQ (n = 6 customers)

### Given Data
Interarrival times: A1=0.4, A2=1.2, A3=0.5, A4=1.7, A5=0.2, A6=1.6, A7=0.2, A8=1.4, A9=1.9
Service times: S1=2.0, S2=0.7, S3=0.2, S4=1.1, S5=3.7, S6=0.6

Arrival times (cumulative sums of Aᵢ):
| Customer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Aᵢ | 0.4 | 1.2 | 0.5 | 1.7 | 0.2 | 1.6 | 0.2 | 1.4 |
| Arrival time | 0.4 | 1.6 | 2.1 | 3.8 | 4.0 | 5.6 | 5.8 | 7.2 |

**Goal:** Simulate until n = 6 customers have completed their delays.

### Event-by-Event Trace

Each event box in the slides shows a **Server box** (idle, or holding the customer currently in service) and a **Queue box** (FIFO list of waiting customers, front-to-back).

**e0 — Initialization (t = 0):**
- Clock = 0, Server status = 0 (idle)
- Number in queue = 0, Arrival list = {}
- Time of last event = 0
- Event list: Next arrival = 0 + A1 = 0.4; Next departure = ∞
- Counters: # delayed = 0, Total delay = 0, Area Q(t) = 0, Area B(t) = 0
- Server=idle, Queue=(empty). Departure=∞ means "not scheduled" → forces next event to be arrival at t=0.4.

**e1 — Arrival of Customer 1 (t = 0.4):**
- Area updates (Δt = 0.4−0 = 0.4): Area Q(t) += 0×0.4 = 0; Area B(t) += 0×0.4 = 0 (using OLD values, both were 0)
- Time of last event: 0 → 0.4
- C1 enters service immediately (server was idle). Server = [C1]. Queue = (empty).
- Schedule: Next arrival = 0.4+1.2 = 1.6; Next departure = 0.4+2.0 = 2.4
- 1.6 < 2.4 ⇒ next event: arrival at t=1.6

**e2 — Arrival of Customer 2 (t = 1.6):**
- Areas (Δt = 1.6−0.4 = 1.2): Q(t) was 0 → area += 0×1.2 = 0; B(t) was 1 → area += 1×1.2 = 1.2
- Server busy → C2 joins queue. Server=[C1], Queue=[C2]
- Schedule: Next arrival = 1.6+0.5 = 2.1; Departure unchanged at 2.4
- Event list: Arr 2.1, Dep 2.4 → next: arrival at 2.1

**e3 — Arrival of Customer 3 (t = 2.1):**
- Clock 1.6→2.1. Server still busy. C3 joins queue behind C2. # in queue: 1→2. Arrival list = {1.6, 2.1}
- Areas (Δt=0.5): Q: +1×0.5=0.5 (running total: 0.5); B: +1×0.5=0.5 (running total: 1.7)
- Server=[C1], Queue=[C2, C3]
- Event list: Arr 3.8, Dep 2.4 → next event: **departure at 2.4** (first departure!)

**e4 — Departure of Customer 1 (t = 2.4):**
- Areas (Δt=0.3, using OLD values Q=2, B=1): Q: +2×0.3=0.6 (total: 1.1); B: +1×0.3=0.3 (total: 2.0)
- Critical ordering: area uses OLD Q(t)=2, THEN we decrease to 1. Compute D2 before advancing the arrival list.
- Queue: 2→1. Arrival list = {2.1}. C2 enters service: S2=0.7 → departure = 2.4+0.7 = 3.1
- Server=[C2], Queue=[C3]

**e5 — Departure of Customer 2 (t = 3.1):**
- Clock 2.4→3.1. C2 departs. Queue has C3 → C3 enters service.
- D3 = 3.1 − 2.1 = 1.0. Total delay: 1.8. # delayed: 3.
- S3 = 0.2 → departure = 3.1+0.2 = 3.3
- Areas (Δt=0.7): Q: +1×0.7=0.7 (total: 1.8); B: +1×0.7=0.7 (total: 2.7)
- Server=[C3], Queue=(empty)

**e6 — Departure of C3 (t = 3.3) — Server Goes Idle!**
- Why ∞? If we left a finite departure time, the timing routine would try to "depart" a nonexistent customer. Setting dep = ∞ forces the next event to be an arrival.
- Areas (Δt=0.2): Q: +0; B: +1×0.2=0.2. Totals: Q=1.8, B=2.9.
- Server = idle, Queue = (empty)
- Event list: Arr 3.8, Dep ∞ → next: arrival at 3.8

**e7 — Arrival of Customer 4 (t = 3.8):**
- C4 arrives to find server idle! (Like C1.) D4 = 0. # delayed: 4. Total delay still 1.8.
- Server: 0→1. Departure = 3.8+1.1 = 4.9 (S4=1.1)
- Areas (Δt=0.5, server was idle → B(t)=0): Q:+0; B:+0. Totals unchanged: Q=1.8, B=2.9.
- Server=[C4], Queue=(empty)
- Event list: Arr 4.0, Dep 4.9

**Events e8–e13 (summary table):**

| e | t | Type | Q (before→after) | Delay | #Del |
|---|---|---|---|---|---|
| e8 | 4.0 | Arr C5 | 0→1 | — | 4 |
| e9 | 4.9 | Dep C4 | 1→0 | D5=0.9 | 5 |
| e10 | 5.6 | Arr C6 | 0→1 | — | 5 |
| e11 | 5.8 | Arr C7 | 1→2 | — | 5 |
| e12 | 7.2 | Arr C8 | 2→3 | — | 5 |
| e13 | 8.6 | Dep C5 | 3→2 | D6=3.0 | 6 |

At e13, Server=[C6], Queue=[C7, C8].

**Simulation stops:** Number delayed reaches n=6. T(6) = 8.6. Main program invokes the report generator.

### Computing the Final Results
1. **Average delay in queue:**
   ```
   d̂(6) = (D1+D2+D3+D4+D5+D6)/6 = (0+0.8+1.0+0+0.9+3.0)/6 = 5.7/6 = 0.95
   ```
   (Note: D1=0, D2=0.8 [=2.4−1.6], D3=1.0 [=3.1−2.1], D4=0, D5=0.9, D6=3.0 — matches the earlier mini-example.)
2. **Time-average number in queue:** total area under Q(t) = 9.9 (accumulated event by event) → q̂(6) = 9.9/8.6 = **1.15**
3. **Server utilization:** total area under B(t) = 7.7 (busy from 0.4–3.3 and 3.8–8.6) → û(6) = 7.7/8.6 = **0.90** (90% busy)

**Caveat:** These are estimates from ONE run with only n=6 customers. In practice, simulate n=1000+ and run multiple independent replications. The method is exactly the same — just more events.

### The Clock–Event-List Mechanism
At the end of each event, the **timing routine** scans the event list, finds the smallest time, advances the clock to that time, and hands control to the appropriate event routine. The clock doesn't tick — it **jumps**.

Full timeline of this example: `0 → 0.4 → 1.6 → 2.1 → 2.4 → 3.1 → 3.3 → 3.8 → 4.0 → 4.9 → 5.6 → 5.8 → 7.2 → 8.6` — **13 events, 13 jumps.** Everything between events is skipped (nothing changes there).

### Order of Updates Matters! (recap — memorize for exam)
1. First: update area accumulators (using OLD values of Q(t) and B(t))
2. Then: update state variables (server status, queue length, arrival list)
3. Then: update time-of-last-event to current clock value
4. Then: update event list (schedule new arrivals/departures)

Common errors: updating Q(t) before computing area (wrong height); updating time-of-last-event before computing areas (wrong width); forgetting to set departure to ∞ when queue empties (phantom departures).

### Discrete-Time vs. Continuous-Time Statistics
- **Discrete-time statistic:** average delay d̂(n) — based on a discrete collection of values D1,...,Dn; simple arithmetic average. "Discrete-time" because index i=1,2,... is discrete.
- **Continuous-time statistics:** q̂(n) and û(n) — based on functions of continuous time Q(t) and B(t); computed as area under curve ÷ total time; these are **time-weighted averages**.
- **Why the distinction matters:** you can't just average queue lengths observed at event times — that's biased. A queue of length 5 lasting 10 minutes matters more than length 5 lasting 0.1 seconds; must weight by duration.

### Summary of the Hand-Simulation Method
1. Initialize the system (empty and idle)
2. Repeatedly find the next event and advance the clock
3. At each event, update state, counters, and areas — **in the right order**
4. Schedule new future events as needed
5. Stop when a condition is met (here: 6 delays observed)
6. Compute performance measures from accumulated statistics

---

## 7. Steps in a Simulation Study

**Big picture:** Simulation is not just programming — model programming is just part of the effort. Must also attend to modeling system randomness, validation, statistical analysis, and project management. **A simulation study is not a simple sequential process** — as one proceeds, it may be necessary to go back to a previous step.

### The 10-Step Flowchart
```
 1. Formulate problem and plan the study
              |
              v
 2. Collect data and define a model  <----------------+
              |                                        |
              v                                        |
     < Assumptions document valid? > --No--------------+
              |
             Yes
              v
 4. Construct a computer program and verify
              |
              v
 5. Make pilot runs
              |
              v
     < Programmed model valid? > --No----(back to step 2, same arrow as above)
              |
             Yes
              v
 7. Design experiments
              |
              v
 8. Make production runs
              |
              v
 9. Analyze output data
              |
              v
10. Document, present, and use results
```
(Note: there is no separately-numbered "step 3" box in the flow — step 3 IS the "Assumptions document valid?" decision diamond after step 2; step 6 IS the "Programmed model valid?" decision diamond after step 5.)

### Step 1: Formulate Problem and Plan the Study
(a) Problem of interest stated by the manager — may not be stated correctly/quantitatively; iterative process often needed to pin it down.
(b) One or more **kickoff meetings** with project manager, simulation analysts, and subject-matter experts (SMEs). Discussed at kickoff: overall objectives; specific questions to be answered (these determine required model detail!); performance measures for evaluating configurations; scope of the model; system configurations to be modeled (determines program generality); time frame and required resources.
(c) Select software for the model:
| | Simulation software (Arena, Flexsim, ProModel) | General-purpose language (C, C++, Java, Python) |
|---|---|---|
| | Faster development, lower project cost | More control, possibly faster execution |
**Key point:** the specific questions to be answered determine required model detail — don't decide the detail first.

### Step 2: Collect Data and Define a Model
(a) Collect info on system structure/operating procedures — no single person/document is sufficient; some people give inaccurate info, identify true SMEs; operating procedures may not be formalized.
(b) Collect data to specify model parameters and input probability distributions.
(c) Delineate all info/data in a written **assumptions document**.
(d) Collect data on performance of the existing system (for validation in step 6).
(e) Choosing level of model detail is an art, depends on: project objectives/performance measures; data availability/credibility concerns; computer constraints; opinions of SMEs; time and money constraints.
(f) There should NOT be a one-to-one correspondence between each model element and each real-system element.
(g) Start with a simple model, embellish as needed — modeling every aspect is seldom required for effective decisions; over-modeling → excessive execution time, missed deadlines, obscured important factors.
(h) Interact with manager and key personnel regularly.

### Step 3: Is the Assumptions Document Valid?
- Perform a **structured walk-through** of the assumptions document before an audience of managers, analysts, and SMEs.
- Purpose: ensure assumptions correct/complete; promote interaction among project members; promote ownership of the model.
- Takes place BEFORE programming begins, to avoid significant reprogramming later.
- **If "No":** go back to step 2 — collect more data, refine model, revise document. Do NOT proceed to programming with invalid assumptions.

### Step 4: Construct a Computer Program and Verify
(a) Program the model in a programming language (C/C++/Java) or simulation software (Arena, ExtendSim, Flexsim, ProModel).
| | Programming language | Simulation software |
|---|---|---|
| Pros | Known, more control, low cost, faster execution | Less programming effort, lower project cost |
| Cons | More coding effort | Less flexibility |
(b) **Verify (debug)** the simulation program. **Verification means: does the program do what we intended?**
Techniques: code review and structured walk-through; **trace** (print detailed event-by-event output, check by hand); check output for reasonableness; run under simplifying assumptions where the answer is known.
**Remember:** Verification ≠ validation. **Verification:** program matches the model. **Validation:** model matches reality.

### Step 5: Make Pilot Runs
- Pilot runs are made for validation purposes in step 6 — preliminary, not final production runs.
- Use output to check whether model behaves as expected; compare against historical data if available.
- Analogy: "test drive / driving around the block" before a 1000-km road trip.

### Step 6: Is the Programmed Model Valid?
(a) If an existing system exists, compare model and system performance measures for the existing configuration.
(b) Regardless, simulation analysts and SMEs should review model results for correctness.
(c) Use **sensitivity analyses** to determine which model factors significantly impact performance measures (and thus must be modeled carefully).
**If "No":** go back to step 2 (all the way, not just the previous step) — because invalid output often means the conceptual model or data is wrong, not just the code.

### Step 7: Design Experiments
Specify, for each system configuration of interest:
- **Length of each simulation run** — how much simulated time needed for accurate estimates
- **Length of the warmup period**, if appropriate — many systems start "empty and idle" (realistic, e.g. bank opening); may need to discard initial transient data
- **Number of independent simulation runs** using different random numbers — enables confidence intervals and precision quantification

### Step 8: Make Production Runs
The "real" runs, per the step-7 design. Output from these runs is used for decision-making. Unlike pilot runs, production runs use the full experimental design: proper run lengths, warmup periods, multiple replications.

### Step 9: Analyze Output Data
Two major objectives:
1. Determine **absolute** performance of a configuration ("what is the expected waiting time for this design?")
2. Compare alternative configurations in a **relative** sense ("Is design A better than B?" "Which of 5 designs is best?")

**Statistical rigor:** simulation outputs are random (inputs are random) — cannot decide based on a single number from a single run. Need: **confidence intervals** ("95% confident avg wait is between 3.2–4.1 min") and **statistical tests** ("difference between A and B is statistically significant"). Treating simulation output as deterministic is a common and serious error.

### Step 10: Document, Present, and Use Results
(a) Document: the assumptions (step 2); the computer program (so others can use/modify); the study's results (for current/future projects). Critical even if "done" — future analysts will thank you (or curse you).
(b) Present results: use animation to communicate to managers/non-experts; discuss model building and validation process to promote credibility.
(c) Results are used in decision-making if both **valid** (technical property) and **credible** (social property — decision-makers must trust results enough to act on them).

### The Two Validation Checkpoints (diagram)
```
Step 2: Data & model  <---------------------+
      |                                     |
      v                                     |
 < Assumptions valid? > --No-----------------+
      |                                     |
     Yes                                    |
      v                                     |
Steps 4-5: Code & pilot                     |
      |                                     |
      v                                     |
 < Model valid? > --No-----------------------+
      |
     Yes
      v
Steps 7-10: Experiment
```
**Both "No" arrows go back to Step 2** — not to the immediately previous step. By design: if validation fails, the problem is usually in the conceptual model or the data, not just the code.

---

## 8. Verification and Validation (V&V)

**Why V&V matters:** the most beautiful simulation model is useless if it doesn't accurately represent reality. Simulations can incorporate any level of detail, making them appear realistic on the surface — this apparent realism can fool you. A wrong model, convincingly animated, gives false confidence ("a GPS that confidently drives you off a cliff").

### Goals of V&V (twofold)
1. **Accuracy:** produce a model that represents true system behavior closely enough to substitute for the actual system (experimenting, analyzing, predicting).
2. **Credibility:** increase confidence in the model to an acceptable level so managers/decision-makers will actually use the results.

**Key insight:** Validation is not an isolated set of procedures following model development — it is an integral part of model development, conducted throughout the process.

### The Model-Building Process (3 steps, revisited iteratively)
1. **Observe the real system:** collect data, talk to operators/technicians/engineers/supervisors/managers
2. **Construct a conceptual model:** a collection of assumptions
3. **Implement an operational model:** translate the conceptual model into simulation code

As development proceeds, new questions arise and you return to earlier steps.

**Conceptual Model** = collection of:
1. **Component assumptions:** entities, resources, their properties
2. **Structural assumptions:** how components interact, what simplifications/abstractions are made
3. **Data assumptions:** hypotheses about model input parameter values and distributions governing random variables

**Operational Model** = the computer program implementing these assumptions in simulation software.

### Three Key Concepts
| Verification | Validation |
|---|---|
| "Did we build the model right?" | "Did we build the right model?" |
| Comparing conceptual model to computer representation | Confirming the model accurately represents the real system |
| Is the model implemented correctly? Are input parameters and logical structure represented correctly? | Does the model's behavior match reality? |

**Calibration:** the iterative engine that drives validation. The process of comparing the model to the real system, identifying discrepancies, revising the model, comparing again, repeating until accuracy is judged acceptable. Validation is usually achieved through calibration.

### The V&V Framework (triangle diagram)
```
                     Real System
                    /            \
        Conceptual                Calibration &
        validation                validation
                  /                    \
    Conceptual Model  <-Model verif.->  Operational Model
    1. Component assumptions            (Computer code)
    2. Structural assumptions
    3. Data assumptions
```
This is an **iterative process** — the model builder goes around this triangle many times, continually comparing the real system to both the conceptual and operational models, modifying each to improve accuracy.

### Verification Techniques
The purpose: ensure the conceptual model is accurately reflected in the operational model (are abstractions/simplifications correctly represented in the code?). These are "common-sense" software-engineering practices:
1. **Independent code review:** someone other than the developer checks the model, ideally an expert in the simulation software.
2. **Flow diagrams:** include each logically possible action for each event type; follow model logic for every action of every event.
3. **Examine output for reasonableness:** run under varied input settings; display and closely examine a wide variety of output statistics.
4. **Check input parameters:** print input parameters at end of simulation to ensure they weren't inadvertently changed during execution.
5. **Self-documentation:** precise definition of every variable, description of every submodel/procedure/major code section — helps others (or yourself later) verify logic/completeness.
6. **Animation:** if the model is animated, watch it — do things look right?
7. **Interactive Run Controller (IRC) / Debugger:** advance simulation to a desired time/condition then display info; focus on a particular entity/code line/procedure and pause when active; observe values of variables/attributes/queues/resources/counters at pause points; temporarily suspend simulation to reassign values or redirect entities.
8. **Graphical interfaces:** essentially a form of self-documentation, simplifies understanding.

**Recommendation:** of all these, the first two (code review, flow diagrams) plus documentation should ALWAYS be done. Close examination of output for reasonableness is especially valuable.

### Checking Output Reasonableness
The easiest, most valuable, and most often overlooked verification technique. **Before** running the model, forecast a reasonable range for selected output statistics — reduces the temptation to rationalize a discrepancy and fail to investigate. Example: in a queueing network, even if mainly interested in response time, also collect utilizations, avg queue lengths, waiting times.

### Verification by Trace
A **trace** = detailed, timestamped simulation output whenever an event occurs (values of selected system states, entity attributes, model variables). Purpose: verify correctness by hand-calculating and comparing against trace output.
Practical considerations: trace over a large time span produces enormous output — restrict to a short period; ensure each event type occurs at least once; for rare events, use artificial data to force occurrence (legitimate for verification purposes). Many tools support selective traces (triggered at specific locations/entities/conditions, e.g. "turn on trace when queue > 5").

**Worked example — Catching a Bug with a Trace:** A single-server queue run over 16 time units gave L_Q(hat) = 0.4375 customers, which seemed reasonable. But the analyst traced anyway:

| CLOCK | EVTYP | NCUST | STATUS |
|---|---|---|---|
| 0 | Start | 0 | 0 (Idle) |
| 3 | Arrival | 1 | 0 (Idle) ← **Bug!** |
| — | Depart | 0 | — |
| 5 | Arrival | 1 | 0 (Idle) |
| 11 | Arrival | 2 | 0 (Idle) |
| 12 | Depart | 1 | 1 (Busy) |
| 16 | — | — | 1 (Busy) |

At CLOCK=3: 1 customer present but server idle — impossible, server should start serving immediately. The bug: `L_Q(hat) = [(1−0)×2 + (0−0)×6 + (1−0)×1 + (2−1)×4] / 16 = 7/16 = 0.4375`.
**Moral:** the output measure had a reasonable value and was computed *correctly* from the data, but the value was wrong because STATUS never assumed correct values. The trace caught an error that summary statistics missed entirely.

### Calibration and Validation of Models
Calibration and validation are conceptually distinct but usually conducted simultaneously.
- **Validation:** the overall process of comparing the model and its behavior to the real system and its behavior.
- **Calibration:** the iterative process of (1) comparing model to real system, (2) making adjustments, (3) comparing revised model to reality, (4) making additional adjustments, (5) repeating until accuracy is sufficient.

Comparison carried out via:
- **Subjective tests:** knowledgeable people judge the model and its output
- **Objective tests:** require data on system behavior and corresponding model data, compared using statistical tests

**The Calibration Loop (diagram):**
```
Initial model --compare--> Real system
     | revise
     v
First revision --compare again--> Real system
     | revise
     v
Second revision --compare again--> Real system
     |
     ⋮
```
**Validation is NOT binary:** no model is ever 100% valid; each revision costs time/money. The real question: is it accurate enough for the decisions we need to make?

### Calibration and Validation Data
Possible criticism: "You've just fitted the model to one specific dataset!" To alleviate:
1. Collect system data
2. Split into two sets: **calibration set** (build/tune model) and **validation set** (reserved for final testing)
3. Calibrate the model using the first set
4. Test the calibrated model against the second set
If unacceptable discrepancies found in "final" validation: return to calibration phase, modify model until acceptable. This shows the model works on unseen data, not just data it was tuned on.

### The Naylor–Finger (1967) Three-Step Approach
A widely followed validation framework:
1. **Face validity:** build a model that appears reasonable to knowledgeable people
2. **Validate model assumptions:** are structural and data assumptions correct?
3. **Validate input-output transformations:** does the model produce outputs consistent with the real system's outputs?

#### Step 1: Face Validity
Show your model to people who know the real system; ask "does this seem right?" Potential users should be involved from conceptualization to implementation: they ensure reasonable structural assumptions, supply reliable data, catch deficiencies, evaluate output reasonableness. Also increases perceived validity/credibility (without which managers won't trust results) and gives users ownership of the model. Subjective but extremely valuable.

**Sensitivity analysis for face validity:** ask the model user "does the model behave as expected when an input variable changes?" E.g., increase arrival rate → do utilizations/queue lengths/delays increase? Add a server → do delays decrease? If the model behaves counter-intuitively, something may be wrong. If real data exist for ≥2 parameter settings, objective statistical sensitivity tests can be conducted.

#### Step 2: Validating Model Assumptions
Two classes:
- **Structural assumptions:** how is the system organized? (one queue or many? do customers switch lines? fixed or variable # servers? queue discipline?) Verify via actual observation during appropriate time periods and discussions with managers/operators.
- **Data assumptions:** what distributions govern the random variables? Three-step statistical analysis: (1) identify an appropriate probability distribution, (2) estimate parameters of the hypothesized distribution, (3) validate via a goodness-of-fit test (chi-square, Kolmogorov–Smirnov) and graphical methods.

**Data reliability:** cross-check with multiple sources (operators may have inaccurate info); when combining data collected at different times, test for homogeneity first (do both sets come from the same population?); test for correlation in the data — only proceed with standard analysis once assured of a random sample.

#### Step 3: Validating Input-Output Transformations
Test of the model as a whole — the model is an input-output transformation:
```
f(X, D) = Y
```
- **X:** uncontrollable input variables (e.g., arrival times, service times)
- **D:** decision variables / controllable parameters (e.g., number of servers)
- **Y:** output/response variables (e.g., utilization, delay)
- **f:** the transformation

The question: when the model receives the same (or statistically equivalent) inputs as the real system, does it produce outputs consistent with the real system's outputs? Requirement: some version of the system must exist so system data can be collected. If the system is still in planning with no operating data, complete input-output validation is impossible.

### Validating Models of New/Modified Systems
Often the model compares alternative designs or investigates behavior under new input conditions. If the existing-system model has been validated, can that confidence transfer to a model of a proposed (nonexistent) system? **Generally yes** — if the new model is a relatively minor modification of the old one.

Changes ranging minor → major:
1. Minor: change a single parameter (machine speed, arrival rate, # servers)
2. Minor: change the form of a distribution (service time distribution)
3. Major: change logical structure of a subsystem (queue discipline, scheduling rule)
4. Major: completely different system design (e.g., computerized inventory replacing manual)

For minor changes (1–2): verify carefully, accept output with considerable confidence.
For major changes (3–4): partial validation may be possible if similar subsystems exist elsewhere.
**There is no way to completely validate the input-output transformations of a model of a nonexisting system.**
