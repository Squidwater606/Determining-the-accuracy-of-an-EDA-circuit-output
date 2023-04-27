# imports:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Critical-Damping - Pulse, Rectangle, Sin & Triangle Waves:
criticalDampingFilepaths = ['data/signalData/critically_damped/CDPULSE.csv','data/signalData/critically_damped/CDRECT.csv','data/signalData/critically_damped/CDSIN.csv','data/signalData/critically_damped/CDTRI.csv'] 
cdγ,cdω = 1.5,0.5

# Overdamping - Pulse, Rectangle, Sin & Triangle Waves:
overDampingFilepaths = ['data/signalData/overdamped/ODPULSE.csv','data/signalData/overdamped/ODRECT.csv','data/signalData/overdamped/ODSIN.csv','data/signalData/overdamped/ODTRI.csv']
odγ,odω = 2.0,0.5

# Underdamping - Pulse, Rectangle, Sin & Triangle Waves:
underDampingFilepaths = ['data/signalData/underdamped/UDPULSE.csv','data/signalData/underdamped/UDRECT.csv','data/signalData/underdamped/UDSIN.csv','data/signalData/underdamped/UDTRI.csv']
udγ,udω = 0.5,0.5

columnIndex = ['in s','C1 in V','C2 in V','C3 in V','C4 in V']
scalings = [-1,-10,-100]

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
    return np.fft.rfft(func,n=shape),np.fft.rfftfreq(time.shape[0],timeStep(time))

def ifft(ft,shape):
    if np.any(np.isnan(ft)) or np.any(np.isinf(ft)):
        ft[np.isnan(ft) | np.isinf(ft)] = 1e-15
    return np.fft.irfft(ft,n=shape)

def convolve(func1,func2,time):
    func1Fft,func2Fft = fft(func1,time)[0],fft(func2,time)[0]
    ftConvolve = func1Fft*func2Fft
    return ifft(ftConvolve,time.shape[0])

def normalizedCrossCorrelation(func1, func2,time):
    C12 = convolve(func1,func2,time)
    normFactor = np.sqrt(np.sum(func1**2)*np.sum(func2**2))
    normCrossCorr = C12 / normFactor
    return normCrossCorr,np.max(normCrossCorr)

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


def plot(filepaths,columnIndex,γ,ω,scalings):
    fig, axs = plt.subplots(nrows=5, ncols=len(filepaths), figsize=(20, 10))
    for i, filepath in enumerate(filepaths):
        data = readData(filepath)
        if data is not None:
            time,inputSignal,sumAmpSignal,invAmpSignal,integAmp2Signal  = data[columnIndex[0]],data[columnIndex[1]],data[columnIndex[2]],data[columnIndex[3]],data[columnIndex[4]]
            timeZeroed = time - time[0]
            solvedODE = solveODEs(inputSignal, time, γ, ω)
            y,yPrime,yDoublePrime = solvedODE[0][::-1],solvedODE[1][::-1],solvedODE[2][::-1]
            crossCorr,crossCorrCoef = normalizedCrossCorrelation(integAmp2Signal*-1,y[::1],time)
            axs[0][i].plot(timeZeroed, inputSignal,color='c')
            axs[1][i].plot(timeZeroed,y,color='r')
            axs[2][i].plot(timeZeroed,yPrime,color='r')
            axs[3][i].plot(timeZeroed,yDoublePrime,color='r')
            axs[1][i].plot(timeZeroed,integAmp2Signal*scalings[0],color='m')
            axs[2][i].plot(timeZeroed,invAmpSignal*scalings[1],color='m')
            axs[3][i].plot(timeZeroed,sumAmpSignal*scalings[2],color='m')
            axs[4][i].plot(timeZeroed,crossCorr,color='g')
            axs[4][i].text(0.95, 0.95, f"Normalized 'r': {crossCorrCoef:.2f}", transform=axs[4][i].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
    fig.text(0.5, 0.04, 'Time (s)', ha='center')
    fig.text(0.04, 0.5, 'Voltage (V)', va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

# Critical-Damping:
plot(criticalDampingFilepaths,columnIndex,cdγ,cdω,scalings)

# Overdamping:
plot(overDampingFilepaths,columnIndex,odγ,odω,scalings)

# Underdamping:
plot(underDampingFilepaths,columnIndex,udγ,udω,scalings)