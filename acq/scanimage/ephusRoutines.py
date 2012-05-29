import numpy as np
import scipy
import glob

def parseXSG(filename):
    raw = scipy.io.loadmat(filename, squeeze_me=True)

    header = raw['header']
    data = raw['data']
    
    acq_fields = data.dtype.names
    header_fields = header.dtype.names

    xsgDict = {}

    for i, field in enumerate(acq_fields):
        xsgDict[field] = {}

    xsgDict['sampleRate'] = header['acquirer'][()]['acquirer'][()]['sampleRate'][()][()]
    xsgDict['epoch'] = header['xsg'][()]['xsg'][()]['epoch'][()][()]
    xsgDict['acquisitionNumber'] = header['xsg'][()]['xsg'][()]['acquisitionNumber'][()][()]

    try:
        ephys_trace_fields = [i for i in data['ephys'][()].dtype.names if 'trace' in i]
    except TypeError: #no traces
        print "No ephys traces?"
        ephys_trace_fields = []
    for chan in ephys_trace_fields:
        xsgDict['ephys'][u'chan0'] = data['ephys'][()][chan][()]

    try:
        acq_trace_fields = [i for i in data['acquirer'][()].dtype.names if 'trace' in i]
    except TypeError: #no traces
        print "No acquirer traces?"
        acq_trace_fields = []

    acq_channel_field_names = [i for i in data['acquirer'][()].dtype.names if 'channelName' in i]
    acq_chan_names = [data['acquirer'][()][i][()][()] for i in acq_channel_field_names]
    for chan, chanName in zip(acq_trace_fields, acq_chan_names):
        xsgDict['acquirer'][chanName] = data['acquirer'][()][chan][()]

    return xsgDict

def parseAllXSGFiles(listOfFilenames, epoch=None):
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
        for i in range(len(xsg_files)):
            data['acquirer'][acq_channel][:,i] = xsg_files[1]['acquirer'][acq_channel].copy()

    # ephys channels
    for ephys_channel in ephys_chans:
        data['ephys'][ephys_channel] = np.zeros((xsg_files[0]['ephys'][ephys_channel].shape[0], len(xsg_files)))
        for i in range(len(xsg_files)):
            data['ephys'][acq_channel][:,i] = xsg_files[1]['ephys'][ephys_channel].copy()

    return data
