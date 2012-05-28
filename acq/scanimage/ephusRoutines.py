import numpy as np
import scipy
import glob

def parseXSG(filename):
    raw = scipy.io.loadmat(filename, squeeze_me=True)

    header = raw['header']
    data = raw['data']

    
    acq_fields = data.dtype.names
    header_fields = header.dtype.names

    number_of_acq_types = len(acq_fields)
    number_of_header_fields = len(header_fields)

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
        xsgDict['ephys']['chan0'] = data['ephys'][()][chan][()]

    try:
        acq_trace_fields = [i for i in data['acquirer'][()].dtype.names if 'trace' in i]
    except TypeError: #no traces
        print "No acquirer traces?"
        acq_trace_fields = []
    acq_chanName_fields = [i for i in data['acquirer'][()].dtype.names if 'channelName' in i]
    for chan, chanName in zip(acq_trace_fields, acq_chanName_fields):
        xsgDict['acquirer'][chanName] = data['acquirer'][()][chan][()]

    return xsgDict

def parseAllXSGFiles(dirname, epoch=None):
    if isinstance(epoch, int):
        epoch = unicode(epoch)

    fileNames = glob.glob(dirname+'/*.xsg')
    all_xsg_files = [parseXSG(i) for i in fileNames]

    if epoch is not None:
        xsg_files = [i for i in all_xsg_files if i['epoch'] == epoch]
    else:
        xsg_files = all_xsg_files

    data = {}
    data['ephys'] = []
    data['acquirer'] = []

    try:        
        data['ephys'].append(np.array([i['ephys']['chan0'] for i in xsg_files]))
    except KeyError:
        print "No phys chan0, aborting phys"
        data['ephys'] = []

    for i in xsg_files[0]['acquirer'].keys():
        data['acquirer'].append([x['acquirer'][i] for x in xsg_files])
        
    return data
