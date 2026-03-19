## EA framework
The evolutionary algorithm selected is the CMA-ES framework from pycma. It might not be optimal for this challenge but is simpler to handle as a starter EA framework.

## Controller Choice
Since the terrain is simple and flat, we chose the oscillatory controller as it allows the ant to discover smoother and more natural periodic gaits which are suitable in this environment. The neural network controller was promising and might give better performance, but we could not manage to get fast and stable motions at the same time.

## Environment Design
As a reward function we used the initial rewards given, i.e. the forward reward, the survival reward and the control cost.
First, we tuned the weights of these rewards since the ant would not move far forward in this initial configuration. We reduced the weight of the control cost to allow the ant to have more dynamic gaits, and we increased the weight of the forward reward to make it go faster.
Then, we tried to add other reward terms to improve the ant’s dynamics. For example, we added a stagnation term to penalize the ant not moving and a distance term to reward the ant going far from the origin.