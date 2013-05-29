import numpy as np
import scipy
import glob
import datetime

import scipy.io

__all__ = ['parseXSG', 'parseAllXSGFiles', 'parseXSGHeader']

def parseXSG(filename):
    """Function to parse the XSG file format.  Returns a dictionary with epoch string,
    sample rate (assuming equal sample rates on all channels), and data.  Data is stored
    in sub-dictionaries, one for each ephus program (ephys, acquirer, etc).  In turn,
    each of those dictionaries contains a numpy array with the raw values.

    :param: filename: string of .xsg file to parse.
    :returns: dictionary of values as described above
    """
    raw = scipy.io.loadmat(filename, squeeze_me=True)

    header = raw['header']
    data = raw['data']
    
    acq_fields = data.dtype.names
    header_fields = header.dtype.names

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
            for i, pulseName in enumerate(header['stimulator']['stimulator']['pulseNameArray']):
                if header['stimulator']['stimulator']['pulseParameters'][i]['type'][()][()] == 'Literal':
                    xsgDict['stimulator'][pulseName[()]] = header['stimulator']['stimulator']['pulseParameters'][i]['signal'][()][()]
        except:
            print 'no literal pulses?'

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
    raw = scipy.io.loadmat(filename, squeeze_me=True)
    return s2d(raw['header'])

def s2d(s):
    d = {}

    if s.dtype.names is None:
        # if empty
        if s.dtype.name == 'object' and s.ndim is 0:
            return s2d(s[()])
        elif s.ndim is not 0: # then we have a 1d string array or other list
            return [s[i] for i in range(s.shape[0])]
        else:
            return s[()]

    else:
        fields = s.dtype.names
        for field in s.dtype.names:
            d[field] = s2d(s[field][()])
    return d

def matlabDateString2DateTime(dateString):
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

def parseAllXSGFiles(listOfFilenames, epoch=None):
    """Convienence function to parse multiple XSG files in one go.  Takes a list of
    files to read in.  Easiest to generate this using the 'files = !ls *.xsg' command in
    ipython, or the glob.glob module.

    Returns a dictionary with epoch and sample rate, and sub-dictionaries for each ephus
    program (ephys, acquirer, etc).  These sub-dictionaries include 2d numpy arrays for each
    channel acquirered.  The first dimension is the number of sampled points, and the second
    is the number of files passed in.

    :param: listOfFilenames: list of strings of the files to read
    :param: epoch: optional parameter to filter by epoch value.  can be a string or an int.
    :returns: dictionary of data as described above.
    """
    if isinstance(epoch, int):
        epoch = unicode(epoch)

    all_xsg_files = [parseXSG(i) for i in listOfFilenames]

    if epoch is not None:
        xsg_files = [i for i in all_xsg_files if i['epoch'] == epoch]
    else:
        xsg_files = all_xsg_files

    data = {}
    # we're assuming that the files are consistant-
    # same channels, same sample rates, etc
    data['epoch'] = xsg_files[0]['epoch']
    data['sampleRate'] = xsg_files[0]['sampleRate']
    data['acquirer'] = {}
    data['ephys'] = {}
    
    acq_chans = xsg_files[0]['acquirer'].keys()
    ephys_chans = xsg_files[0]['ephys'].keys()

    # acq channels
    for acq_channel in acq_chans:
        data['acquirer'][acq_channel] = np.zeros((xsg_files[0]['acquirer'][acq_channel].shape[0], len(xsg_files)))
        try:
            for i in range(len(xsg_files)):
                data['acquirer'][acq_channel][:,i] = xsg_files[i]['acquirer'][acq_channel].copy()
        except ValueError:
            print "Acquirer trace size mis-match.  Make sure settings didn't change from file to file- ensure files are same all same size."
            return None
            
    # ephys channels
    for ephys_channel in ephys_chans:
        data['ephys'][ephys_channel] = np.zeros((xsg_files[0]['ephys'][ephys_channel].shape[0], len(xsg_files)))
        try:
            for i in range(len(xsg_files)):
                data['ephys'][acq_channel][:,i] = xsg_files[i]['ephys'][ephys_channel].copy()
        except ValueError:
            print "Ephys trace size mis-match.  Make sure settings didn't change from file to file- ensure files are same all same size."
            return None

    return data
