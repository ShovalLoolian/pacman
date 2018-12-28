from typing import Any

from layout import getLayout
from pacman import *
from submission import *
from ghostAgents import *
from textDisplay import *
from matplotlib import pyplot as plt
from util import Counter
import csv

if __name__ == '__main__':

    vec_org = [0,0,0,0,0]
    vec_reflex = [0,0,0,0,0]
    vec_minmax = [0,0,0,0,0]
    vec_alphabeta = [0,0,0,0,0]
    vec_random = [0,0,0,0,0]

    vec_org_time = [0,0,0,0,0]
    vec_reflex_time = [0,0,0,0,0]
    vec_minmax_time = [0,0,0,0,0]
    vec_alphabeta_time = [0,0,0,0,0]
    vec_random_time = [0,0,0,0,0]

    with open('experiments.csv') as file_ptr:
        # csv_reader = file_ptr.read(csv_file)
        for rowssss in file_ptr:
            # print(row)
            row = rowssss.split(",")
            agent = row[0]
            d = int(row[1])
            score = float(row[3])
            time = float(row[4])
            if agent == 'OriginalReflexAgent':
                vec_org[d] += score
                vec_org_time[d] += time
            if agent == 'ReflexAgent':
                vec_reflex[d] += score
                vec_reflex_time[d] += time
            if agent == 'MinimaxAgent':
                vec_minmax[d] += score
                vec_minmax_time[d] += time
            if agent == 'AlphaBetaAgent':
                vec_alphabeta[d] += score
                vec_alphabeta_time[d] += time
            if agent == 'RandomExpectimaxAgent':
                vec_random[d] += score
                vec_random_time[d] += time
        file_ptr.close()

    x = [0, 1, 2, 3, 4]

    # score
    plt.figure(1)
    plt.plot([1], vec_org[1:2],'o', label="OriginalReflexAgent")
    plt.plot([1], vec_reflex[1:2], 'o', label="ReflexAgent")
    plt.plot([2,3,4], vec_minmax[2:5], label="MinimaxAgent")
    plt.plot([2,3,4], vec_alphabeta[2:5], label="AlphaBetaAgent")
    plt.plot([2,3,4], vec_random[2:5], label="RandomExpectimaxAgent")

    plt.xlabel("Depth")
    plt.ylabel("Sum of average score")
    plt.title("Score as a function of depth")
    plt.legend()
    plt.grid()
    plt.show()

    # time
    plt.figure(2)
    plt.plot([1], vec_org_time[1:2], 'o', label="OriginalReflexAgent")
    plt.plot([1], vec_reflex_time[1:2], 'o', label="ReflexAgent")
    plt.plot([2, 3, 4], vec_minmax_time[2:5], label="MinimaxAgent")
    plt.plot([2, 3, 4], vec_alphabeta_time[2:5], label="AlphaBetaAgent")
    plt.plot([2, 3, 4], vec_random_time[2:5], label="RandomExpectimaxAgent")

    plt.xlabel("Depth")
    plt.ylabel("Sum of time")
    plt.title("Time as a function of depth")
    plt.legend()
    plt.grid()
    plt.show()
    exit()