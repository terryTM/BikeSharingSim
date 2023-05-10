import numpy as np
import pandas as pd
import scipy.stats as stats



def bikeShare(n, m, lam, p, q, mu, sigma, totTime, maxBikes = None):
    if maxBikes == None:
        return 0

    interarrival = np.random.exponential(1/lam, n)
    interarrival = np.cumsum(interarrival)
    #print(interarrival)

    #stations = [[] for i in range(m)]  #Store departure times for each station
    
    stock = [[0 for _ in range(maxBikes)] for _ in range(m)]
    

    #print(stock)
    cumWait = 0
    servedRiders = 0
    waiting_riders = [[] for _ in range(m)]

    for rider in range(n):
        #print("Rider Num")
        #print(rider)

        arrivalTime = interarrival[rider]
        #print("Rider Arrival Time")
        #print(arrivalTime)

        if arrivalTime > totTime: #Stop simulation when time limit is reached
            break

        servedRiders += 1 #Counts the number of riders who rode before simulation cut off (24hrs)
        mystation = np.random.choice(range(m), p = p)
        #print("Chosen Stations")
        #print(mystation)
        
        destination_station = np.random.choice(range(m), p = q[mystation])
        #print("Destination Stations")
        #print(destination_station)
        waiting = True
        usage = np.random.lognormal(mu, sigma)

        #print("Usage Time")
        #print(usage)


        # Remove bikes that have already been returned before the rider's arrival time
        #stations[mystation] = [t for t in stations[mystation] if t > arrivalTime]
        for i in range(len(stock[mystation])):
            #If there is avaliable stock we remove that bike from the station and append its arrival time to destination station
            if stock[mystation][i] < arrivalTime:
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

                waiting_riders[destination_station].pop(0)
                
        

        
        #print(stock)
        #print("Wait Time")
        #print(wait_time)
        #print("------------------------------")
            
        cumWait += wait_time

    
    #If there are still riders waiting at the end of simulated riders we will append that to the total wait

    #for i in waiting_riders:
    #    if (len(i) > 0):
    #        for j in i:
    #            cumWait += (totTime - j[0])

    # Prob succesful bike rental
    count = 0
    for i in waiting_riders:
        for j in i:
            count += 1

    prob_success = (servedRiders - count) / (servedRiders)

    

    avg_wait_time = cumWait / servedRiders

    return avg_wait_time, prob_success
#Params


df_p = pd.read_csv('/Users/terryma/Documents/start_station_probs.csv')



df = pd.read_csv('/Users/terryma/Documents/trip_stats.csv')
unique_stations = set(df['start'].unique().tolist() + df['end'].unique().tolist())
unique_stations = sorted(unique_stations)
stat1 = unique_stations
stat2 = df_p['start_station_name'].tolist()
#print(len(stat1))
#print(len(stat2))


diff_list1 = [x for x in stat1 if x not in stat2]

streets = diff_list1

new_data = {'start_station_name': streets, 'probability': [0] * len(streets)}
new_df = pd.DataFrame(new_data)
result = pd.concat([df_p, new_df], ignore_index=True)

df_p = result.sort_values(by='start_station_name')
p = df_p['probability'].tolist()
#print(p)

#print(df_p)


all_combinations = pd.DataFrame([(s1, s2) for s1 in unique_stations for s2 in unique_stations], columns=['start', 'end'])

merged_df = all_combinations.merge(df, on=['start', 'end'], how='left')
merged_df['count'] = merged_df['count'].fillna(0)

sum_counts = merged_df.groupby('start')['count'].sum().reindex(unique_stations, fill_value=0)
merged_df['probability'] = merged_df.apply(lambda row: row['count'] / sum_counts[row['start']], axis=1)

q_matrix = merged_df.pivot_table(index='start', columns='end', values='probability', fill_value=0)
q_matrix = q_matrix.reindex(index=unique_stations, columns=unique_stations, fill_value=0)

q = q_matrix.values.tolist()
print(len(q), len(q[0]))

#unique_stations_from_q = q_matrix.index.values
#all_stations_p = pd.DataFrame({'start_station_name': unique_stations_from_q})
#merged_p = all_stations_p.merge(df_p, on='start_station_name', how='left')
#merged_p['probability'] = merged_p['probability'].fillna(0)
#merged_p['probability'] = merged_p['probability'] / merged_p['probability'].sum()
#p = merged_p['probability'].tolist()



print("LEN P: ", len(p))

n = 3500
totTime = 1440  # 24 hours in minutes
m = len(p)
lam = 2.38
mu = 2.78
sigma = 0.619
maxBikes = 10  # 10 bikes per station

wait_times = []
prob_successes = []


num_simulations = 50
confidence_level = 0.9


for _ in range(num_simulations):
    wait, prob_success = bikeShare(n, m, lam, p, q, mu, sigma, totTime, maxBikes)
    wait_times.append(wait)
    prob_successes.append(prob_success)

mean_wait_time = np.mean(wait_times)
std_wait_time = np.std(wait_times)
mean_prob_success = np.mean(prob_successes)
std_prob_success = np.std(prob_successes)
t_critical = stats.t.ppf((1 + confidence_level) / 2, num_simulations - 1)
wait_time_margin_of_error = t_critical * (std_wait_time / np.sqrt(num_simulations))
prob_success_margin_of_error = t_critical * (std_prob_success / np.sqrt(num_simulations))

wait_time_confidence_interval = (mean_wait_time - wait_time_margin_of_error, mean_wait_time + wait_time_margin_of_error)
prob_success_confidence_interval = (mean_prob_success - prob_success_margin_of_error, mean_prob_success + prob_success_margin_of_error)

print("90% Conf for prob succesful rental:", prob_success_confidence_interval)
print("90% Conf for wait time:", wait_time_confidence_interval)


