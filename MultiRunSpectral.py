import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Argument parser for CLI interaction Jul 29
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)

parser.add_argument(
    "--dataset", type=str, default=None, help="Dataset to use. available are IndianPines, Salinas, PaviaC and PaviaU"
)
parser.add_argument(
    "--training_sample", type=float, default=0.95,  help="train-test split, defaults to 0.95. Cannot be higher than 0.99"
)
parser.add_argument(
    "--patchSize", type=int, default=24, help="Patch size(?) defaults to 24"
)
parser.add_argument(
    "--epoch", type=int, default=100, help="Number of epochs defaults to 100"
)
parser.add_argument(
    "--runs", type=int, default=1, help="Number of runs defaults to 1"
)

args = parser.parse_args()

DATASET = args.dataset
TEST_RATIO = args.training_sample
WINDOWSIZE = args.patchSize
EPOCH = args.epoch
N_RUNS = args.runs

results = []
# run the experiment several times
for run in range(N_RUNS):
    os.system("python SpectralNet.py --dataset "+str(DATASET)+" --epoch "+str(EPOCH)+" --test_ratio "+str(TEST_RATIO)+" --windowSize "+str(WINDOWSIZE))
    with open('myOutput.txt', 'r') as input:
        temp = []
        for line in input:
            print(run)
            print(line) 
            temp.append(float(line.strip()))
        results.append(temp)
print(results)
averageResults = np.mean(results, axis=0),np.std(results,axis=0)
print(averageResults)
f = open("experimentResults.txt","w")
f.write("Dataset: " + DATASET + "\n")
f = open("experimentResults.txt","a")
f.write("Training Percentage: " +str(TEST_RATIO)+"\n")
f.write("Number of Runs: "+str(N_RUNS)+"\n")
f.write("Number of Epochs: "+str(EPOCH)+"\n")
f.write("Kappa: {:.03f} +- {:0.3f}\n".format(averageResults[0][0],averageResults[1][0]))
f.write("Overall accuracy: {:.03f} +- {:0.3f}\n".format(averageResults[0][1],averageResults[1][1]))
f.write("Average accuracy: {:.03f} +- {:0.3f}\n".format(averageResults[0][2],averageResults[1][2]))
f.write("Training time: {:.03f} +- {:0.3f}\n".format(averageResults[0][3],averageResults[1][3]))
f.write("Testing time: {:.03f} +- {:0.3f}\n".format(averageResults[0][4],averageResults[1][4]))
f.write("Total time: {:.03f} +- {:0.3f}\n".format(averageResults[0][5],averageResults[1][5]))
f = open("experimentResults.txt","r")
print(f.read())
f.close()
