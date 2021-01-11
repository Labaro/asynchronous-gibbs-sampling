from .control import Control
from .toy_problem import ToyProblem
import numpy as np

if __name__ == "__main__":

    transmission_rate = 0.075
    max_delay = 1000
    nMC = 10000
    mixing_rate = []
    time = []
    message_stat = []
    for _ in range(3):
        toy_problem = ToyProblem()
        control = Control(4, toy_problem, max_delay, transmission_rate, nMC, cyclic=False)
        control.run()
        rate, t, mess_send, mess_received = control.show_results()
        mixing_rate.append(rate)
        time.append(t)
        message_stat.append([mess_send, mess_received])
        with open("data.txt","a") as f:
            f.write(str(mixing_rate[-1]) + ";" + str(time[-1]) + ";" + str(message_stat[-1]) + "\n")
    print("Average of the mixing rate:", np.mean(np.max(mixing_rate, axis=1)))
    print("Average time:", np.mean(time))
    avg_mess = np.mean(message_stat, axis=0)
    avg_mess_send, avg_mess_receive = avg_mess
    print("Messages send:", avg_mess_send, "Percentage of message received:", avg_mess_receive/avg_mess_send)

