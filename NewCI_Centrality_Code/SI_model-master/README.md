# SI_model
Simulating how infection spreads in US airport network. 

Task is to implement an susceptible-infected (SI) disease spreading
model and run it against time-stamped air transport data. Especially, we will investigate how
static network properties can be used to understand disease spreading.

In the implementation of the SI-model, there is initially only one infected airport and all other
airports are susceptible. A susceptible node may become infected, when an airplane originating
from an infected node arrives to the airport. (Note that, for the airplane to carry sick passengers,
the source airport needs to be infected at the departure time of the flight.) If this condition is
met, the destination airport becomes infected with probability p ∈ [0, 1] reflecting the infectivity
of the disease. Nodes that have become infected remain infected indefinitely.
