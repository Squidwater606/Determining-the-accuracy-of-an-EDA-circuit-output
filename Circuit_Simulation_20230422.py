# imports:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Critical-Damping - Pulse, Rectangle, Sin, Sinc & Triangle Waves:
criticalDampingFilepaths = ['useable_data/critically_damped/CDPULSE.csv','useable_data/critically_damped/CDRECT.csv','useable_data/critically_damped/CDSIN.csv','useable_data/critically_damped/CDSINC.csv','useable_data/critically_damped/CDTRI.csv'] 
cdγ,cdω = 1.5,0.5

# Overdamping - Pulse, Rectangle, Sin, Sinc & Triangle Waves:
overDampingFilepaths = ['useable_data/overdamped/ODPULSE.csv','useable_data/overdamped/ODRECT.csv','useable_data/overdamped/ODSIN.csv','useable_data/overdamped/ODSINC.csv','useable_data/overdamped/ODTRI.csv']
odγ,odω = 2.0,0.5

# Underdamping - Pulse, Rectangle, Sin, Sinc & Triangle Waves:
underDampingFilepaths = ['useable_data/underdamped/UDPULSE.csv','useable_data/underdamped/UDRECT.csv','useable_data/underdamped/UDSIN.csv','useable_data/underdamped/UDSINC.csv','useable_data/underdamped/UDTRI.csv']
udγ,udω = 0.5,0.5

columnIndex = ['in s','C1 in V','C2 in V','C3 in V','C4 in V']

# define functions:
def readData(path):
    try:
        data = pd.read_csv(path, delimiter=',')
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

def timeStep(time): 
    return np.average(np.diff(time))

def fft(func,time): 
    shape = time.shape[0]
    if np.any(np.isnan(func)) or np.any(np.isinf(func)):
        func[np.isnan(func) | np.isinf(func)] = 1e-15
    return np.fft.rfft(func,n=shape,norm='ortho'),np.fft.rfftfreq(time.shape[0],timeStep(time))

def ifft(ft,shape):
    if np.any(np.isnan(ft)) or np.any(np.isinf(ft)):
        ft[np.isnan(ft) | np.isinf(ft)] = 1e-15
    return np.fft.irfft(ft,n=shape,norm='ortho')

def solveODEs(func,time,γ,ω):
    try:
        ftFunc,freqs = fft(func,time)
        shape = time.shape[0]
        yFft = ftFunc/(-(freqs**2)-(1j)*γ*(freqs)+ω)
        y = ifft(yFft,shape)
        yPrime = np.gradient(y,time)
        yDoublePrime = np.gradient(yPrime,time)
        return y,yPrime,yDoublePrime
    except AttributeError:
        print("Invalid argument passed to solveODE")
        return None
    except ValueError:
        print("Invalid input value for solveODE")
        return None
    except Exception as e:
        print(f"An error occurred in solveODE: {e}")
        return None

def plotODEs(filepaths,columnIndex,γ,ω):
    fig, axs = plt.subplots(nrows=4, ncols=len(filepaths), figsize=(20, 10))
    for i, filepath in enumerate(filepaths):
        data = readData(filepath)
        if data is not None:
            func, time = data[columnIndex[1]], data[columnIndex[0]]
            solvedODE = solveODEs(func, time, γ, ω)
            axs[0][i].plot(time, func)
            axs[0][i].set_title(filepath.split('/')[-1].split('.')[0])
            axs[1][i].plot(time, solvedODE[0])
            axs[2][i].plot(time, solvedODE[1])
            axs[3][i].plot(time, solvedODE[2])
    fig.text(0.5, 0.04, 'Time (s)', ha='center')
    fig.text(0.04, 0.5, 'Voltage (V)', va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

def plotData(filepaths,columnIndex):
    fig, axs = plt.subplots(nrows=4, ncols=len(filepaths), figsize=(20, 10))
    for i, filepath in enumerate(filepaths):
        data = readData(filepath)
        if data is not None:
            inputSignal,sumAmpSignal,invAmpSignal,integAmp2Signal,time = data[columnIndex[1]],data[columnIndex[2]],data[columnIndex[3]],data[columnIndex[4]],data[columnIndex[0]]
            axs[0][i].plot(time, inputSignal)
            axs[0][i].set_title(filepath.split('/')[-1].split('.')[0])
            axs[1][i].plot(time, integAmp2Signal)
            axs[2][i].plot(time, invAmpSignal)
            axs[3][i].plot(time, sumAmpSignal)
    fig.text(0.5, 0.04, 'Time (s)', ha='center')
    fig.text(0.04, 0.5, 'Voltage (V)', va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

# Critical-Damping:
plotODEs(criticalDampingFilepaths,columnIndex,cdγ,cdω)
plotData(criticalDampingFilepaths,columnIndex)

# Overdamping:
plotODEs(overDampingFilepaths,columnIndex,odγ,odω)
plotData(overDampingFilepaths,columnIndex)

# Underdamping:
plotODEs(underDampingFilepaths,columnIndex,udγ,udω)
plotData(underDampingFilepaths,columnIndex)