import heapq
from queue import Queue
interarrival = [0.4,1.2,0.5,1.7,0.2,1.6,0.2,1.4,1.9] 
service = [2.0,0.7,0.2,1.1,3.7,0.6]

event_list = []
heapq.heappush(event_list,(interarrival[0],"A"))

arrival_list = Queue()

print((event_list==[]))
def simulate():
    interarrival_index = 1 
    service_index = 0
    q = 0
    q_len = 0  
    b = 0 
    last_event_time = 0
    is_busy = 0 
    curr_time = interarrival[0]
    delay = 0 
    delay_count = 0                          
    while(not event_list==[]):
        event = heapq.heappop(event_list) 
        curr_time = event[0]
        print(f"current time: {event[0]}")
        elapsed_time = event[0] - last_event_time
        q += q_len*elapsed_time 
        b += is_busy*elapsed_time 
        if(event[1]=="A"):
            print("arrival")
            if(is_busy==0):
                if(q_len!=0):
                    print("qlen not 0, probably an error")
                is_busy=1
                delay += 0                    
                delay_count += 1              
                print(f"DELAY {delay_count} = 0")
                if service_index < len(service):
                    next_service_end_time = curr_time + service[service_index]
                    print(f"next service end:{next_service_end_time}")
                    heapq.heappush(event_list,(next_service_end_time,"D"))
                    service_index += 1 
                else:
                    print("WARNING: ran out of service times, cannot schedule departure")
                if interarrival_index < len(interarrival):
                    next_arrival_time = curr_time + interarrival[interarrival_index]
                    print(f"next arrival time:{next_arrival_time}")
                    heapq.heappush(event_list,(next_arrival_time,"A"))
                    interarrival_index += 1 
                else:
                    print("WARNING: ran out of interarrival times, no more arrivals scheduled")
            else: 
                q_len += 1 
                arrival_list.put(event[0])
                if interarrival_index < len(interarrival):
                    next_arrival_time = curr_time + interarrival[interarrival_index]
                    print(f"next arrival time:{next_arrival_time}")
                    heapq.heappush(event_list,(next_arrival_time,"A"))
                    interarrival_index += 1 
                else:
                    print("WARNING: ran out of interarrival times, no more arrivals scheduled")
        else:
            print("departure")
            if(not arrival_list.empty()):
                wait_started_at = arrival_list.get() 
                delay += (event[0]-wait_started_at) 
                delay_count += 1              
                print(f"DELAY {delay_count} = {event[0]-wait_started_at}")
            if(q_len>0):
                q_len -= 1 
                if service_index < len(service):
                    next_service_end_time = curr_time + service[service_index]
                    heapq.heappush(event_list,(next_service_end_time,"D"))
                    service_index += 1 
                    print(f"next service end:{next_service_end_time}")
                else:
                    print("WARNING: ran out of service times, cannot schedule departure")
            elif(q_len==0):
                is_busy = 0
                # only push the phantom "keep the heap alive" event if a future arrival
                # could still occur -- otherwise there's nothing left to simulate
                if interarrival_index < len(interarrival):
                    next_service_end_time = 1e9
                    heapq.heappush(event_list,(next_service_end_time,"D"))
                    print(f"next service end:{next_service_end_time}")
                else:
                    print("No more arrivals possible and queue is empty -- nothing left to simulate.")
        last_event_time = event[0]
        if(delay_count==6):
            avg_delay = delay/delay_count 
            avg_queue = q/last_event_time 
            utilization = b/last_event_time 
            print(avg_delay,avg_queue,utilization)
            return avg_delay,avg_queue,utilization 

simulate()