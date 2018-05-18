# based on multilineplot example in matplotlib with MRI data (I think)
# uses line collections (might actually be from pbrain example)
# clm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
import math

subtitle_len = 250


def stackplot(marray, fs=256, seconds=None, start_time=None, ylabels=None, show=True, trim=False, autolim=False, dataRange=[]):
    """
    will plot a stack of traces one above the other assuming
    marray.shape = numRows, numSamples
    """
    tarray = np.transpose(marray)
    fig=stackplot_t(tarray, fs=fs, seconds=seconds, start_time=start_time, ylabels=ylabels, show=show, trim=trim, autolim=autolim, dataRange=dataRange)
    return fig


def stackplot_t(tarray, fs=256, seconds=None, start_time=None, ylabels=None, show=True, trim=False, autolim=False, dataRange=[]):
    """
    will plot a stack of traces one above the other assuming
    tarray.shape =  numSamples, numRows
    """
    if trim:
        tarray = tarray[start_time*fs:(start_time+seconds)*fs, :]
    step = int(math.ceil(tarray.shape[0] / float(fs*180)))
    tarray = tarray[range(0, tarray.shape[0], step), :]
    data = tarray
    numSamples, numRows = tarray.shape
# data = np.random.randn(numSamples,numRows) # test data
# data.shape = numSamples, numRows
    if seconds:
        t = seconds * np.arange(numSamples, dtype=float)/numSamples
# import pdb
# pdb.set_trace()
        if start_time:
            t = t+start_time
            xlm = (start_time, start_time+seconds)
        else:
            xlm = (0,seconds)

    else:
        t = np.arange(numSamples, dtype=float)
        xlm = (0,numSamples)

    ticklocs = []
    if show:
        plt.ion()
    else:
        plt.ioff()
    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot(111)
    plt.xlim(*xlm)
    # xticks(np.linspace(xlm, 10))
    if dataRange:
        dmin = dataRange[0]
        dmax = dataRange[1]
    else:
        dmin = data.min()
        dmax = data.max()
    dr = (dmax - dmin)*0.7  # Crowd them a bit.
    y0 = dmin
    y1 = (numRows-1) * dr + dmax + subtitle_len
    plt.ylim(y0, y1)

    segs = []
    for i in range(numRows):
        segs.append(np.hstack((t[:,np.newaxis], data[:,i,np.newaxis])))
        # print "segs[-1].shape:", segs[-1].shape
        ticklocs.append(i*dr)

    ticklocs.reverse()

    offsets = np.zeros((numRows,2), dtype=float)
    offsets[:,1] = ticklocs

    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

    lines = LineCollection(segs, offsets=offsets,
                           transOffset=None, colors=colors
                           )

    ax.add_collection(lines, autolim=autolim)

    # set the yticks to use axes coords on the y axis
    ax.set_yticks(ticklocs)
    # ax.set_yticklabels(['PG3', 'PG5', 'PG7', 'PG9'])
    if not plt.ylabel:
        plt.ylabel = ["%d" % ii for ii in range(numRows)]
    ax.set_yticklabels(ylabels)

    plt.xlabel('zaman (sn)')

    return ax

def stackplot_my(tarray, fs=256, seconds=None, start_time=None, ylabels=None, trim=False, autolim=False, dataRange=[], channels=[], my_figure=[]):

    tarray = np.transpose(tarray)

    if channels:
        tarray = tarray[:, channels]

    if trim:
        tarray = tarray[start_time*fs:(start_time+seconds)*fs, :]
    step = int(math.ceil(tarray.shape[0] / float(fs*600)))
    tarray = tarray[range(0, tarray.shape[0], step), :]
    data = tarray

    numSamples, numRows = tarray.shape

    if seconds:
        t = seconds * np.arange(numSamples, dtype=float)/numSamples
        if start_time:
            t = t+start_time
            xlm = (start_time, start_time+seconds)
        else:
            xlm = (0,seconds)

    else:
        t = np.arange(numSamples, dtype=float)
        xlm = (0,numSamples)

    ticklocs = []

    my_figure.set_xlim(*xlm)
    if dataRange:
        dmin = dataRange[0]
        dmax = dataRange[1]
    else:
        dmin = data.min()
        dmax = data.max()
    dr = (dmax - dmin)*0.7  # Crowd them a bit.
    y0 = dmin
    y1 = (numRows-1) * dr + dmax + subtitle_len
    my_figure.set_ylim(y0, y1)

    segs = []
    for i in range(numRows):
        segs.append(np.hstack((t[:,np.newaxis], data[:,i,np.newaxis])))
        ticklocs.append(i*dr)

    ticklocs.reverse()

    offsets = np.zeros((numRows,2), dtype=float)
    offsets[:,1] = ticklocs

    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

    lines = LineCollection(segs, offsets=offsets,
                           transOffset=None, colors=colors
                           )

    my_figure.add_collection(lines, autolim=autolim)

    # set the yticks to use axes coords on the y axis
    my_figure.set_yticks(ticklocs)
    # ax.set_yticklabels(['PG3', 'PG5', 'PG7', 'PG9'])
    if not my_figure.get_ylabel():
        my_figure.set_ylabel = ["%d" % ii for ii in range(numRows)]
    my_figure.set_yticklabels(ylabels)

    my_figure.set_xlabel('time (s)')

    return my_figure


def test_stacklineplot():
    numSamples, numRows = 800, 5
    data = np.random.randn(numRows, numSamples)  # test data
    stackplot(data, 10.0)