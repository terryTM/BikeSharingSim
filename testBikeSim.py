import numpy as np


def bikeShare(n, m, lam, p, q, mu, sigma, totTime, maxBikes = None):
    if maxBikes == None:
        return 0

    interarrival = np.random.exponential(1/lam, n)
    interarrival = np.cumsum(interarrival)
    print(interarrival)

    #stations = [[] for i in range(m)]  #Store departure times for each station
    
    stock = [[0 for _ in range(maxBikes)] for _ in range(m)]
    

    print(stock)
    cumWait = 0
    servedRiders = 0
    waiting_riders = [[] for _ in range(m)]

    for rider in range(n):
        print("Rider Num")
        print(rider)

        arrivalTime = interarrival[rider]
        print("Rider Arrival Time")
        print(arrivalTime)

        if arrivalTime > totTime: #Stop simulation when time limit is reached
            break

        servedRiders += 1 #Counts the number of riders who rode before simulation cut off (24hrs)
        mystation = np.random.choice(range(m), p = p)
        print("Chosen Stations")
        print(mystation)
        
        destination_station = np.random.choice(range(m), p = q[mystation])
        print("Destination Stations")
        print(destination_station)
        waiting = True
        usage = np.random.lognormal(mu, sigma)

        print("Usage Time")
        print(usage)


        # Remove bikes that have already been returned before the rider's arrival time
        #stations[mystation] = [t for t in stations[mystation] if t > arrivalTime]
        for i in range(len(stock[mystation])):
            #If there is avaliable stock we remove that bike from the station and append its arrival time to destination station
            if stock[mystation][i] < arrivalTime:
                print("LESS")
                stock[mystation].pop(i)
                waiting = False
                wait_time = 0

                totT = arrivalTime + usage

                stock[destination_station].append(totT)
                break

        #If there are no bikes avaliable we remove the next arriving bike and add to waiting time
        if waiting == True and len(stock[mystation]) > 0:
            soonestRdy = min(stock[mystation])
            wait_time = soonestRdy - arrivalTime
            soonestRdyIdx = stock[mystation].index(soonestRdy)
            stock[mystation].pop(soonestRdyIdx)

            totT = soonestRdy + usage
            stock[destination_station].append(totT)
        
        if (len(stock[mystation]) == 0):
            waiting_riders[mystation].append([arrivalTime, destination_station])

        else:
            if len(waiting_riders[destination_station]) > 0:

                wait_time += totT - waiting_riders[destination_station][0][0]
                simusage = np.random.lognormal(mu, sigma)
                stock[waiting_riders[destination_station][0][1]].append(totT + simusage)
                #print("Stock Before Pop")
                #print(stock)
                stock[destination_station].pop(0)
                waiting_riders[destination_station].pop(0)
                
        

        
        print(stock)
        print("Wait Time")
        print(wait_time)
        print("------------------------------")
            
        cumWait += wait_time

    
    #If there are still riders waiting at the end of simulated riders we will append that to the total wait

    for i in waiting_riders:
        if (len(i) > 0):
            for j in i:
                cumWait += (totTime - j[0])

    

    avg_wait_time = cumWait / servedRiders

    return avg_wait_time
#Params

n = 50  # rider num
totTime = 1440 #24hrs in minutes
m = 2     # station num
lam = 1 #arrival rate mean
p = [0.8, 0.2]
q = [[0.5, 0.5], [0.5, 0.5]]
mu = 1.5  #Mean Usage
sigma = 0.5  #Standard Deviation of Usage
maxBikes = 10
#Run Bike Sim

wait = bikeShare(n, m, lam, p, q, mu, sigma, totTime, maxBikes)
print(wait)


