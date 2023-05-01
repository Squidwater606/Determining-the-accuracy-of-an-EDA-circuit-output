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

# note that ω represents the square natural frequency

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
    shape = len(time)
    if np.any(np.isnan(func)) or np.any(np.isinf(func)):
        func[np.isnan(func) | np.isinf(func)] = 1e-15
    ft,freqs = np.fft.rfft(func,n=shape),np.fft.rfftfreq(len(time),timeStep(time))
    return ft,freqs

def ifft(ft,shape):
    if np.any(np.isnan(ft)) or np.any(np.isinf(ft)):
        ft[np.isnan(ft) | np.isinf(ft)] = 1e-15
    iFt = np.fft.irfft(ft,n=shape)
    return iFt

def crossCorr(ft1,ft2,time):
    ftConvolve1 = np.conj(ft1)*ft2
    ftConvolve2 = np.conj(ft2)*ft1
    convolve1 = ifft(ftConvolve1,len(time))
    convolve2 = ifft(ftConvolve2,len(time))
    return convolve1,convolve2

def normCrossCorr(ft1,ft2,func1,func2,time):
    C12,C21 = crossCorr(ft1,ft2,time)
    normFactor = np.sqrt(np.sum(func1**2)*np.sum(func2**2))
    normC12 = C12/normFactor
    normC21 = C21/normFactor
    nccCoeff = (np.max(np.abs(normC12))+np.max(np.abs(normC21)))/2
    phaseDiff = np.arccos(np.sum(normC21*normC12)/(np.sqrt(np.sum(normC21**2))*np.sqrt(np.sum(normC12**2))))
    return normC12,normC21,nccCoeff,phaseDiff

def solve(sigInpt,func,time,γ,ω):
    shape = len(time)
    ftInpt,freqs = fft(sigInpt,time)
    ftFunc,f = fft(func,time)
    yFft = ftInpt/(-(freqs**2)+γ/2*(1j)*freqs+ω)
    y = ifft(yFft,shape)
    yPrime = np.gradient(y, time)
    yDoublePrime = np.gradient(yPrime, time)
    normC12,normC21,nccCoeff,phaseDiff = normCrossCorr(ftFunc,yFft,func,y,time)
    return y,yPrime,yDoublePrime,normC12,normC21,nccCoeff,phaseDiff

def plotFull(filepaths,columnIndex,γ,ω,scalings):
    fig, axs = plt.subplots(nrows=5, ncols=len(filepaths), figsize=(20, 10))
    for i, filepath in enumerate(filepaths):
        data = readData(filepath)
        if data is not None:
            time,inputSignal,sumAmpSignal,invAmpSignal,integAmp2Signal = data[columnIndex[0]],data[columnIndex[1]],data[columnIndex[2]],data[columnIndex[3]],data[columnIndex[4]]
            timeZeroed = time - time[0]
            solved = solve(inputSignal,integAmp2Signal,time,γ,ω)
            y,yPrime,yDoublePrime,normC12,normC21,nccCoeff,phaseDiff = solved[0],solved[1],solved[2],solved[3],solved[4],solved[5],solved[6]
            axs[0][i].plot(timeZeroed, inputSignal,color='c')
            axs[1][i].plot(timeZeroed,y,color='r')
            axs[2][i].plot(timeZeroed,yPrime,color='r')
            axs[3][i].plot(timeZeroed,yDoublePrime,color='r')
            axs[1][i].plot(timeZeroed,integAmp2Signal*scalings[0],color='g')
            axs[2][i].plot(timeZeroed,invAmpSignal*scalings[1],color='g')
            axs[3][i].plot(timeZeroed,sumAmpSignal*scalings[2],color='g')
            axs[4][i].plot(timeZeroed,normC12,color='m')
            axs[4][i].plot(timeZeroed,normC21,color='y')
            axs[1][i].text(0.95, 0.95, f"Phase-Difference: {phaseDiff:.2f}", transform=axs[1][i].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
            axs[4][i].text(0.95, 0.95, f"NCC Coefficient 'r': {nccCoeff:.2f}", transform=axs[4][i].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
    fig.text(0.5, 0.04, 'Time (s)', ha='center')
    fig.text(0.04, 0.5, 'Voltage (V)', va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

def plotShrunk(filepaths,columnIndex,γ,ω,scalings):
    fig, axs = plt.subplots(nrows=3, ncols=len(filepaths), figsize=(20, 10))
    for i, filepath in enumerate(filepaths):
        data = readData(filepath)
        if data is not None:
            time,inputSignal,integAmp2Signal = data[columnIndex[0]],data[columnIndex[1]],data[columnIndex[4]]
            timeZeroed = time - time[0]
            solved = solve(inputSignal,integAmp2Signal,time,γ,ω)
            y,normC12,normC21,nccCoeff,phaseDiff = solved[0],solved[3],solved[4],solved[5],solved[6]
            axs[0][i].plot(timeZeroed, inputSignal,color='c')
            axs[1][i].plot(timeZeroed,y,color='r')
            axs[1][i].plot(timeZeroed,integAmp2Signal*scalings[0],color='g')
            axs[2][i].plot(timeZeroed,normC12,color='m')
            axs[2][i].plot(timeZeroed,normC21,color='y')
            axs[1][i].text(0.95, 0.95, f"Phase-Difference: {phaseDiff:.2f}", transform=axs[1][i].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
            axs[2][i].text(0.95, 0.95, f"NCC Coefficient 'r': {nccCoeff:.2f}", transform=axs[2][i].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
    fig.text(0.5, 0.04, 'Time (s)', ha='center')
    fig.text(0.04, 0.5, 'Voltage (V)', va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

# full figures:
plotFull(criticalDampingFilepaths,columnIndex,cdγ,cdω,scalings)
plotFull(overDampingFilepaths,columnIndex,odγ,odω,scalings)
plotFull(underDampingFilepaths,columnIndex,udγ,udω,scalings)

# shrunk figures:
plotShrunk(criticalDampingFilepaths,columnIndex,cdγ,cdω,scalings)
plotShrunk(overDampingFilepaths,columnIndex,odγ,odω,scalings)
plotShrunk(underDampingFilepaths,columnIndex,udγ,udω,scalings)