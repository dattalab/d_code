import numpy as np
import scipy
import datetime
import copy

import scipy.io

__all__ = ['parseXSG', 'mergeXSGs', 'parseXSGHeader']

def parseXSG(filename):
    """Function to parse the XSG file format.  Returns a dictionary
    with epoch string, sample rate (assuming equal sample rates on all
    channels), and data.  Data is stored in sub-dictionaries, one for
    each ephus program (ephys, acquirer, etc).  In turn, each of those
    dictionaries contains a numpy array with the raw values.

    There is a lot going on here to deal with the fact that different
    programs can be active, and in particular, stimulator pulses can
    be specified in different ways.  In the end, we have numpy arrays
    of what was sent out and what was recorded.

    There are a couple of fields that seem superfluous, but they are
    to ensure compatibility for extracellular analysis routines from
    spike sort.

    :param: filename: string of .xsg file to parse.
    :returns: dictionary of values as described above
    """
    raw = scipy.io.loadmat(filename, squeeze_me=True)

    header = raw['header']
    data = raw['data']
    
    acq_fields = data.dtype.names

    xsgDict = {}

    for i, field in enumerate(acq_fields):
        xsgDict[field] = {}
    xsgDict['stimulator'] = {}

    header = s2d(header)

    xsgDict['sampleRate'] = int(header['acquirer']['acquirer']['sampleRate'])
    xsgDict['epoch'] = int(header['xsg']['xsg']['epoch'])
    xsgDict['acquisitionNumber'] = header['xsg']['xsg']['acquisitionNumber']
    xsgDict['xsgName'] = filename
    xsgDict['xsgExperimentNumber'] = header['xsg']['xsg']['experimentNumber']
    xsgDict['date'] = matlabDateString2DateTime(header['xsgFileCreationTimestamp'])
    xsgDict['dateString'] = header['xsgFileCreationTimestamp']

    # import ephys traces if needed
    try:
        ephys_trace_fields = [i for i in data['ephys'][()].dtype.names if 'trace' in i]
        # loop over channels (should just be 'chan0' and 'chan1')
        for index, chan in enumerate(ephys_trace_fields):
            xsgDict['ephys'][u'chan'+str(index)] = data['ephys'][()][chan][()]
    except TypeError: #no traces
        xsgDict['ephys'] = None
    
    # import acquirer traces if needed    
    try:
        acq_trace_fields = [i for i in data['acquirer'][()].dtype.names if 'trace' in i]
        # loop over channels
        acq_channel_field_names = [i for i in data['acquirer'][()].dtype.names if 'channelName' in i]
        acq_chan_names = [data['acquirer'][()][i][()][()] for i in acq_channel_field_names]
        for chan, chanName in zip(acq_trace_fields, acq_chan_names):
            xsgDict['acquirer'][chanName] = data['acquirer'][()][chan][()]
    except TypeError: #no traces
        xsgDict['acquirer'] = None

    # rebuild stimulation pulses if needed
    # need to do this in two phases
    # 1) for the stimulator program
    # 2) for the ephys program

    # we'll put both in the 'stimulator' field.

    # for the stimulator program
    try:
        if header['stimulator']['stimulator']['startButton']: # stimulator was engaged

            # put all the square pulse stims in 
            try: 
                sampleRate = int(header['stimulator']['stimulator']['sampleRate'])
                traceLength = int(header['stimulator']['stimulator']['traceLength'])

                delay = header['stimulator']['stimulator']['pulseParameters']['squarePulseTrainDelay'] * sampleRate
                offset = header['stimulator']['stimulator']['pulseParameters']['offset'] * sampleRate
                amp = header['stimulator']['stimulator']['pulseParameters']['amplitude']
                ISI = header['stimulator']['stimulator']['pulseParameters']['squarePulseTrainISI'] * sampleRate
                width = header['stimulator']['stimulator']['pulseParameters']['squarePulseTrainWidth'] * sampleRate
                number_of_pulses = int(header['stimulator']['stimulator']['pulseParameters']['squarePulseTrainNumber'])

                stim_array = np.zeros(sampleRate*traceLength) + offset

                for pulse_number in range(number_of_pulses):
                    start = pulse_number * ISI + delay
                    end = start + width
                    stim_array[start:end] = amp
                xsgDict['stimulator'][header['stimulator']['stimulator']['channels']['channelName']] = stim_array
            except:
                print 'no standard pulses?'

            # put all the literal pulses in
            try:
                num_literal_pulses = 0
                for i, pulseName in enumerate(header['stimulator']['stimulator']['pulseNameArray']):
                    if header['stimulator']['stimulator']['pulseParameters'][i]['type'][()] == 'Literal':
                        num_literal_pulses = num_literal_pulses + 1
                        xsgDict['stimulator'][pulseName] = header['stimulator']['stimulator']['pulseParameters'][i]['signal'][()][()]
            except:
                print 'error parsing literal pulses?'
            if num_literal_pulses is 0:
                print 'no literal pulses found'
    except:
        pass
    # stimulation in the ephys program (a command to the amp)
    try:
        if header['ephys']['ephys']['stimOnArray']:
            sampleRate = int(header['ephys']['ephys']['sampleRate'])
            traceLength = int(header['ephys']['ephys']['traceLength'])

            delay = header['ephys']['ephys']['pulseParameters']['squarePulseTrainDelay'] * sampleRate
            offset = header['ephys']['ephys']['pulseParameters']['offset'] * sampleRate
            amp = header['ephys']['ephys']['pulseParameters']['amplitude']
            ISI = header['ephys']['ephys']['pulseParameters']['squarePulseTrainISI'] * sampleRate
            width = header['ephys']['ephys']['pulseParameters']['squarePulseTrainWidth'] * sampleRate
            number_of_pulses = int(header['ephys']['ephys']['pulseParameters']['squarePulseTrainNumber'])

            stim_array = np.zeros(sampleRate*traceLength) + offset

            for pulse_number in range(number_of_pulses):
                start = pulse_number * ISI + delay
                end = start + width
                stim_array[start:end] = amp
            xsgDict['stimulator']['chan0'] = stim_array   # NOTE: hard coded for now for a single ephys channel
            pass
        else:
            pass # no ephys stim was sent out
        pass
    except:
        pass # ephys stim wasn't on. don't modify

    # if there weren't any pulses sent out, then xsgDict['stimulator'] will be empty
    # we then set the stim key to None
    if xsgDict['stimulator'] == {}:
        xsgDict['stimulator'] = None

    return xsgDict

def parseXSGHeader(filename):
    """Routine to extract just the header from an XSG file.  Uses an
    internal recursive function s2d()"""
    raw = scipy.io.loadmat(filename, squeeze_me=True)
    return s2d(raw['header'])

def s2d(s):
    """This routine takes a (possibly nested) MATLAB struct as parsed
    by scipy.io and turns it into a dictionary"""
    d = {}

    if hasattr(s, 'dtype'):
        if s.dtype.names is None:
            # if empty
            if s.dtype.name == 'object' and s.ndim is 0:
                return s2d(s[()])
            elif s.ndim is not 0: # then we have a 1d string array or other list
                return [s[i] for i in range(s.shape[0])]
            else:
                return s[()]
        else:
            for field in s.dtype.names:
                d[field] = s2d(s[field][()])
        return d
    else:
        return s


def matlabDateString2DateTime(dateString):
    """This a simple routine that parses a string from Matlab
    and turns it into a DateTime object"""

    months = {
        'Jan' : 1,
        'Feb' : 2,
        'Mar' : 3,
        'Apr' : 4,
        'May' : 5,
        'Jun' : 6,
        'Jul' : 7,
        'Aug' : 8,
        'Sep' : 9,
        'Oct' : 10, 
        'Nov' : 11,
        'Dec' : 12
        }

    date = dateString.split(' ')[0].split('-')
    time = dateString.split(' ')[1].split(':')

    year =int(date[2])
    month = months[date[1]]
    day = int(date[0])
    hour = int(time[0])
    minute = int(time[1]) 
    second = int(time[2])

    return datetime.datetime(year, month, day, hour, minute, second)

def mergeXSGs(xsg1, xsg2):
    """This routine merges two xsg dictionaries, concatenating every
    numpy array on a new, last axis and combining all non-numpy fields
    into a list.  This routine is only valid on repetitions of the
    same sort of acquisition files, that is, xsgs that had the same
    combo of acquirer, ephys and stimulator settings, including sample
    rates and lengths!

    Importantly, this function can be used with reduce() to easily
    combine multiple XSGs in a well defined manner.  For example:
    
       files = !ls *xsgs
       allXSGList = [parseXSG(f) for f in files]
       all_xsgs = reduce(mergeXSGs, allXSGList)
       epoch3xsgs = reduce(mergeXSGs, [x for x in allXSGList if x['epoch'] == '3'])

    This behavior could be changed to overwrite instead of appending
    to a list, but I'll leave that for the future.

    :param: xsg1 - a single XSG or merged XSG dictionary
    :param: xsg2 - a single XSG dictionary
    :returns: a merged XSG dictionary.
    """
    xsg1 = copy.deepcopy(xsg1)
    xsg2 = copy.deepcopy(xsg2)

    # calculate which keys will be merged via numpy concat
    # and which by appending lists.

    non_numpy_keys = xsg1.keys()
    for prog in ['acquirer', 'ephys', 'stimulator']:
        non_numpy_keys.remove(prog)
    non_numpy_keys.append('merged')

    numpy_keys = ['acquirer', 'ephys', 'stimulator']

    for prog in ['acquirer', 'ephys', 'stimulator']:
        if xsg1[prog] is None or xsg2[prog] is None:
            non_numpy_keys.append(prog)
            numpy_keys.remove(prog)
    
    # remove duplicate keys
    non_numpy_keys = list(set(non_numpy_keys))
    numpy_keys = list(set(numpy_keys))

    # we need to distinguish between 'merged' and 'unmerged' xsgs xsgs
    # have the 'merged' key set to True, otherwise the key/val doesn't
    # exist

    # the reasoning here is based around the fact that an xsg is not
    # merged, a list is an actual value.  if it is, then it is a list
    # of values from previous mergings.  so, we set both xsgs to
    # 'merged' and wrap everything in a list if it wasn't already a
    # merged XSG.  Wrapping in a list makes merging simple addition on
    # a key by key basis

    for x in [xsg1, xsg2]:
        if 'merged' not in x.keys():
            x['merged'] = True
            for key in non_numpy_keys:
                x[key] = [x[key]]

    # actually merge xsgs
    merged_xsg = {}
    for key in non_numpy_keys:
        merged_xsg[key] = xsg1[key] + xsg2[key]
    for key in numpy_keys:
        # for each program, we have dictionaries with keys of channels
        # and vals of numpy arrays. We need to concatenate the numpy arrays
        # so that the last dimension is trials

        merged_xsg[key] = {}
        for channel in xsg1[key].keys():
            dim_diff = xsg1[key][channel].ndim - xsg2[key][channel].ndim 
            if dim_diff is 0: # two unmerged numpy arrays, promote both
                merged_xsg[key][channel] = np.concatenate((xsg1[key][channel][:,np.newaxis], xsg2[key][channel][:,np.newaxis]), axis=1)

            if dim_diff is 1: # xsg1 is merged and one higher dim, promote xsg2's array
                merged_xsg[key][channel] = np.concatenate((xsg1[key][channel], xsg2[key][channel][:,np.newaxis]), axis=1)

            if dim_diff is -1: # xsg2 is merged and one higher dim, promte xsg1's array (won't happen if using reduce)
                merged_xsg[key][channel] = np.concatenate((xsg1[key][channel][:,np.newaxis], xsg2[key][channel]), axis=1)

    return merged_xsg

