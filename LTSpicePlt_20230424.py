import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Critical-Damping - Pulse, Rectangle, Sin & Triangle Waves:
criticalDampingFilepaths = ['data/signalData/critically_damped/CDPULSE.csv','data/signalData/critically_damped/CDRECT.csv','data/signalData/critically_damped/CDSIN.csv','data/signalData/critically_damped/CDTRI.csv'] 
LTcriticalDampingFilepaths = ['data/LTdata/critically_damped/DiffCircuitV4-15-critDamped-final-pulse.txt','data/LTdata/critically_damped/DiffCircuitV4-15-critDamped-final-rect.txt','data/LTdata/critically_damped/DiffCircuitV4-15-critDamped-final-sin-1x18.txt','data/LTdata/critically_damped/DiffCircuitV4-15-critDamped-final-tria.txt']

# Overdamping - Pulse, Rectangle, Sin & Triangle Waves:
overDampingFilepaths = ['data/signalData/overdamped/ODPULSE.csv','data/signalData/overdamped/ODRECT.csv','data/signalData/overdamped/ODSIN.csv','data/signalData/overdamped/ODTRI.csv']
LToverDampingFilepaths = ['data/LTdata/over_damped/DiffCircuitV4-15-overdamped-final-pulse.txt','data/LTdata/over_damped/DiffCircuitV4-15-overdamped-final-rect.txt','data/LTdata/over_damped/DiffCircuitV4-15-overdamped-final-sin-1x18.txt','data/LTdata/over_damped/DiffCircuitV4-15-overdamped-final-tria.txt']

# Underdamping - Pulse, Rectangle, Sin & Triangle Waves:
underDampingFilepaths = ['data/signalData/underdamped/UDPULSE.csv','data/signalData/underdamped/UDRECT.csv','data/signalData/underdamped/UDSIN.csv','data/signalData/underdamped/UDTRI.csv']
LTunderDampingFilepaths =['data/LTdata/under_damped/DiffCircuitV4-15-underdamped-final-pulse.txt','data/LTdata/under_damped/DiffCircuitV4-15-underdamped-final-rect.txt','data/LTdata/under_damped/DiffCircuitV4-15-underdamped-final-sin-1x18.txt','data/LTdata/under_damped/DiffCircuitV4-15-underdamped-final-tria.txt']

signalColumnIndex = ['in s','C1 in V','C2 in V','C3 in V','C4 in V'] # column index for signal data
LTcolumnIndex = ['time','V(p001)','V(n001)','V(n007)','V(n008)'] # column index for LTSpice simulation data
delimiter = [',','\t']

# Define functions:
def readData(path,delimiter): # reads data files
    try:
        data = pd.read_csv(path, delimiter=delimiter)
        return data
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"File is empty: {path}")
        return None
    except pd.errors.ParserError:
        print(f"Error parsing file: {path}")
        return None

def plotSimu(filepaths1,filepaths2,signalColumnIndex,LTcolumnIndex,delimiter):
    fig, axs = plt.subplots(nrows=4, ncols=len(filepaths1), figsize=(20, 10))
    for i, filepath1 in enumerate(filepaths1):
        realData = readData(filepath1,delimiter[0])
        if realData is not None:
            time,inputSignal,sumAmpSignal,invAmpSignal,integAmp2Signal  = realData[signalColumnIndex[0]],realData[signalColumnIndex[1]],realData[signalColumnIndex[2]],realData[signalColumnIndex[3]],realData[signalColumnIndex[4]]
            timeZeroed = time - time[0]
            axs[0][i].plot(timeZeroed, inputSignal,color='r')
            axs[0][i].set_title(filepath1.split('/')[-1].split('.')[0])
            axs[1][i].plot(timeZeroed, integAmp2Signal,color='r')
            axs[2][i].plot(timeZeroed, invAmpSignal,color='r')
            axs[3][i].plot(timeZeroed, sumAmpSignal,color='r')
    for i, filepath2 in enumerate(filepaths2):
        LTdata = readData(filepath2,delimiter[1])
        if LTdata is not None:
            LTtime,LTinputSignal,LTinvAmpSignal,LTsumAmpSignal,LTintegAmp2Signal = LTdata[LTcolumnIndex[0]],LTdata[LTcolumnIndex[1]],LTdata[LTcolumnIndex[2]],LTdata[LTcolumnIndex[3]],LTdata[LTcolumnIndex[4]]
            LTtimeZeroed = LTtime - LTtime[0]
            axs[0][i].plot(LTtimeZeroed,LTinputSignal,color='b')
            axs[1][i].plot(LTtimeZeroed, LTintegAmp2Signal,color='b')
            axs[2][i].plot(LTtimeZeroed, LTinvAmpSignal,color='b')
            axs[3][i].plot(LTtimeZeroed, LTsumAmpSignal,color='b')
    fig.text(0.5, 0.04, 'Time (s)', ha='center')
    fig.text(0.04, 0.5, 'Voltage (V)', va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

plotSimu(criticalDampingFilepaths,LTcriticalDampingFilepaths,signalColumnIndex,LTcolumnIndex,delimiter)
plotSimu(overDampingFilepaths,LToverDampingFilepaths,signalColumnIndex,LTcolumnIndex,delimiter)
plotSimu(underDampingFilepaths,LTunderDampingFilepaths,signalColumnIndex,LTcolumnIndex,delimiter)