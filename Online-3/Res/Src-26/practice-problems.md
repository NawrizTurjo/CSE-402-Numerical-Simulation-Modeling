Here are some practice problems for Monte Carlo approximation, ranging from easy to moderate. Since you have already done basic integration and approximating $\pi$, these problems will introduce new concepts like higher dimensions, expected value, discrete probability, and real-world simulation.

### 1. Estimating the Volume of a 3D Sphere (Easy)
**Concept:** Extending 2D area calculations to 3D volume.
**Problem:** You have a cube centered at the origin $(0,0,0)$ with side length 2 (meaning $x, y,$ and $z$ all range from -1 to 1). Inside this cube is a perfectly centered sphere with a radius of 1. 
**Task:** Write a Monte Carlo simulation to estimate the volume of this sphere.
**Hint:** 
* Generate random points $(x, y, z)$ uniformly in the range $[-1, 1]$.
* Check if the point falls inside the sphere using the formula $x^2 + y^2 + z^2 \le 1$.
* The ratio of points inside the sphere to total points is equal to the ratio of the sphere's volume to the cube's volume.
* **Expected Output:** The volume should be approximately $\frac{4}{3}\pi(1)^3 \approx 4.188$.

### 2. The Expected Value of a Dice Game (Easy-Moderate)
**Concept:** Using Monte Carlo to find the expected value (mean outcome) of a discrete random process.
**Problem:** You are playing a game where you roll 3 standard 6-sided dice. Your score is calculated as follows:
* If all three dice show the same number, you win $\$20$.
* If exactly two dice show the same number, you win $\$5$.
* If all three dice are different, you lose $\$2$.
**Task:** Use a Monte Carlo simulation to estimate the expected value (average money won or lost) per game if you play 100,000 times.
**Hint:** You can simulate a dice roll using a random integer generator between 1 and 6. Simulate 100,000 games, keep a running total of the money won/lost, and divide by 100,000 at the end.

### 3. Estimating the Area Under a Weird Curve (Moderate)
**Concept:** Monte Carlo integration where analytical calculus is difficult.
**Problem:** Estimate the area under the curve $y = \sin(x) \cdot \cos(x^2)$ from $x = 0$ to $x = \pi$.
**Task:** Create a bounding box for this function (you know $x$ goes from $0$ to $\pi$, and since $\sin$ and $\cos$ are bounded by $-1$ and $1$, $y$ will be between $0$ and $1$ for this range). Generate random points in this box and determine the area.
**Hint:** The exact analytical answer to this is quite difficult to calculate by hand, which is exactly where Monte Carlo shines!

### 4. The 1D Random Walk (Moderate)
**Concept:** Simulating stochastic processes (paths that change randomly over time).
**Problem:** A drunk person starts at position 0 on a number line. At each step, they have a 50% chance of moving $+1$ and a 50% chance of moving $-1$. They take exactly 100 steps. 
**Task:** Use a Monte Carlo simulation to estimate the probability that the person ends up **further than 15 steps away from the origin** (i.e., absolute final position > 15) after 100 steps.
**Hint:** 
* Simulate one "walk" of 100 steps by adding 100 random choices of $+1$ or $-1$ together.
* Check if the absolute value of the final position is $> 15$.
* Repeat this walk 50,000 times and count how many times the condition is met. Divide by 50,000 to get the probability.

### 5. Project Completion Time (Moderate - Real World Application)
**Concept:** Monte Carlo simulation for project management (often used in PERT analysis).
**Problem:** You have a project that consists of 3 sequential tasks (Task B cannot start until Task A finishes, etc.).
* Task A takes between 2 and 4 days (uniformly distributed).
* Task B takes between 3 and 6 days (uniformly distributed).
* Task C takes between 1 and 5 days (uniformly distributed).
**Task:** Estimate the probability that the total project takes **more than 11 days** to complete.
**Hint:** 
* For one simulation, pick a random number for A between 2 and 4, a random number for B between 3 and 6, and a random number for C between 1 and 5.
* Add them together. Check if the sum is $> 11$.
* Run this 100,000 times and find the percentage of times the sum exceeded 11.

### 6. Buffon's Needle Problem (Moderate - Classic Probability)
**Concept:** A famous historical Monte Carlo problem to estimate $\pi$ without using circles!
**Problem:** Imagine a floor made of parallel wooden planks, each 1 unit wide. You drop a needle of length $L$ (let's use $L = 0.5$) onto the floor. 
**Task:** Simulate dropping the needle 100,000 times and estimate the probability that the needle crosses a line between two planks. Then, use this probability to estimate the value of $\pi$.
**Hint:**
* To simulate a drop, you need two random numbers:
  1. The distance from the center of the needle to the nearest line (this is uniformly distributed between $0$ and $0.5$).
  2. The angle the needle makes with the horizontal lines (uniformly distributed between $0$ and $\pi/2$).
* The needle crosses a line if: `distance <= (L/2) * sin(angle)`.
* The theoretical probability of crossing is $\frac{2L}{\pi \times \text{width}}$. Therefore, $\pi \approx \frac{2L}{P \times \text{width}}$.

*(Note: If you are using Python, the `numpy` or `random` libraries will be your best friends for these. Let me know if you want the solution code or mathematical explanations for any of these!)*