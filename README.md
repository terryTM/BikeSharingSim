# BikeSharingSim

Implements a discrete-event simulation to study Citi Bike, which is New York City's bike-sharing service. We run this simulator on data from June 2022.

**Implementation:**

First we simulate n rider arrival times using an exponential distribution. We make sure to cut off the riders at time 1440 which is 24 hours in minutes.
Next we set up stock which has the maximum number of bikes at each station there. The number for each station in stock represents the time by which the bike is available. We check if the rider arrives after the bike becomes available which the rider will then check out. The check out process is simply removing the bike from stock of the point of departure and appending that bikes arrival time to the destination station. The arrival time is our usage time plus our time of departure. Our usage time is determined based on our lognormal distribution.
One case we have to consider is if a user arrives and no bikes are available. If there are future bikes that are already coming in we take the next incoming bike and take the arrival time of that bike minus our time of arrival of the rider to get a wait time.
Another case is if there are no incoming bikes coming to a station. We simply append those users to a waiting queue where they will take the next incoming bike.
Finally to verify my code, I printed the stocks, arrival time, rider num, and other stats and verified it to make sure it makes sense.
