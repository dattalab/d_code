# scanimage specific analysis functions

import glob
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import pdb

import scipy.ndimage
import scipy.stats
import scipy.io

import pymongo
import pickle
import pymorph

import imaging.io
import imaging.alignment
import imaging.segmentation
import traces
from DotDict import DotDict


__all__ = ['readRawSIImages', 'parseSIHeaderFile', 'importTrial', 'addLineToStateVar', 'readMultiOdorEpoch', 'addTrialToEpoch',
       'calculateEpochAverages', 'buildCellDataFromEpoch', 'calculateCellDataAmplitudes', 'calculateResponderMasks',
       'baselineCellData', 'normalizeCellData', 'averageCellData', 'plotNormCells', 'plotMeanCells',
       'plotRespondersAndNonResponders', 'plotMeanRespondersAndNonResponders', 'parseXSG']

#### SI specific file i/o

def readRawSIImages(tif_files, txt_files):
    image = imaging.io.imread(tif_files[0])
    nFrames = image.shape[2]

    headerState = parseSIHeaderFile(txt_files[0])
    # grab channel info from the header
    activeChannels = map(int, [headerState['acq']['savingChannel1'],
                               headerState['acq']['savingChannel2'],
                               headerState['acq']['savingChannel3'],
                               headerState['acq']['savingChannel4']])
    nActiveChannels=sum(activeChannels)

    imageSize=image.shape

    imageChannels=np.zeros([4,imageSize[0], imageSize[0], imageSize[2]/nActiveChannels, len(tif_files)])

    files = zip(tif_files,txt_files)
    for index, (tifFileName,headerFileName) in enumerate(files):
        print index
        image = imaging.io.imread(tifFileName)
        nFrames = image.shape[2]

        for chanNum in range(4):
            if (activeChannels[chanNum]):
                imageChannels[chanNum,:,:,:,index] = np.array(image[:,:,chanNum:nFrames:nActiveChannels])

                maxPixelValue                = float(scipy.ndimage.maximum(imageChannels[chanNum][:,:,:,index]))
                imageChannels[chanNum,:,:,:,index] = imageChannels[chanNum][:,:,:,index].astype('float')/maxPixelValue
                imageChannels[chanNum,:,:,:,index] = (imageChannels[chanNum][:,:,:,index]*255).astype('uint8')

    return imageChannels

def parseSIHeaderFile(txtFile):
    # read header file, create dictionary
    state={}
    header_file = open(txtFile, 'r').read().split('\r')
    for line in header_file:
        key,sep,value = line.partition('=')
        keyList = key.split('.')
        value = value.strip('\'\"')
        state=addLineToStateVar(keyList[1:],value,state)
    return state

def addLineToStateVar(keyList,value,state):
    """
    Recursive function builds a hierarchal dictionary from
    a MATLAB style struct definition string, i.e.:

    state.blah.foo = bar

    gets turned into

    state:{blah: {foo:bar}}

    Function has to check if a key exists at that level of the
    dict and if not, must create an empty dict.

    Basic logic: If there are more keys still, recurse, otherwise assign value to key.

    This could probably be improved by using the setdefault method of the dict... but not sure how exactly

    Arguments:
    - `keyList`:LHS of the state line : ['olfactometer','nOdors']
    - `value`: RHS of the state line : '4'
    - `state`:top level of hierarchical dict
    """

    try:
        top=keyList.pop(0)
    except IndexError: # blank line, skip!
        return state

    if not top in state.keys(): # insure the dict exists
        state[top]={}

    if keyList: # do we have deeper to go?  recurse if so
        state[top]=addLineToStateVar(keyList,value,state[top])
    else:
        state[top]=value

    return DotDict(state)


def importTrial(tif_filename, header_filename):
    # returns a list of odor response dictionaries for this trial

    raw_image = imaging.io.imread(tif_filename)
    
    state = parseSIHeaderFile(header_filename)

    # calc base, odor, post frame numbers


    frames = map(int, state['olfactometer']['odorFrameListString'].split(';'))
    frame_states = map(int, state['olfactometer']['odorStateListString'].split(';'))
    frame_states_no_zeros = filter(lambda x: x>0, frame_states)

    pre_frames   = frames[0]
    odor_frames  = frames[1]
    post_frames  = frames[2]
    blank_frames = frames[3]


    single_odor_frame_length = np.sum(frames[0:3])
    single_odor_frame_length_with_blank = np.sum(frames[0:4])

    # make a list of imaging channels acquired

    activeChannels = map(int, [state['acq']['savingChannel1'],
                               state['acq']['savingChannel2'],
                               state['acq']['savingChannel3'],
                               state['acq']['savingChannel4']])

    nActiveChannels=sum(activeChannels)

    # first, let's demultiplex image into it's channels
    # this will be used below

    raw_image_in_channels = {}
    for chanNum in range(4):
        if (activeChannels[chanNum]):
            raw_image_in_channels[chanNum] = raw_image[:,:,chanNum::nActiveChannels].copy()
        else:
            raw_image_in_channels[chanNum] = np.array([])
    

    # get list of odors

    state['olfactometer']['odorStateList'] = state['olfactometer']['odorStateListString'].split(';')
    state['olfactometer']['odorTimeList']  = state['olfactometer']['odorTimeListString'].split(';')
    state['olfactometer']['odorFrameList'] = state['olfactometer']['odorFrameListString'].split(';')

    valve_numbers_as_presented = state['olfactometer']['odorStateList'][1::4]
    odor_list_indicies = map(lambda x: int(x)-1, valve_numbers_as_presented) # annoyingly, valve # is 1-order, not 0 order
    nOdors = state['olfactometer']['nOdors']

    basename = tif_filename[0:-4].split('_')[0]
    epoch = tif_filename[0:-4].split('_')[1][1:]
    acqNum = tif_filename[0:-4].split('_')[2]

    trial = [{} for odor in odor_list_indicies]

    for odor_index in odor_list_indicies:
        trial[odor_index] = {}

        # store meta data
        trial[odor_index]['allHeaderData'] = state

        trial[odor_index]['raw_tif_filename'] = tif_filename

        trial[odor_index]['epochNumber'] = epoch
        trial[odor_index]['baseName'] = basename
        trial[odor_index]['date'] =  state['internal']['startupTimeString']


        trial[odor_index]['resolution'] = (state['acq']['linesPerFrame'] + 'x' +
                                           state['acq']['linesPerFrame'] + 'x' +
                                           state['acq']['msPerLine'] + 'ms')

        trial[odor_index]['baselineFrames'] = [0, pre_frames]
        trial[odor_index]['odorFrames']     = [pre_frames, pre_frames + odor_frames]
        trial[odor_index]['postFrames']     = [pre_frames + odor_frames, pre_frames + odor_frames + post_frames]

        trial[odor_index]['odor1Name'] = state['olfactometer']['valveOdor1Name_'+str(odor_index+1)]
        trial[odor_index]['odor2Name'] = state['olfactometer']['valveOdor2Name_'+str(odor_index+1)]
        trial[odor_index]['odor1Dilution'] = state['olfactometer']['valveOdor1Dilution_'+str(odor_index+1)]
        trial[odor_index]['odor2Dilution'] = state['olfactometer']['valveOdor2Dilution_'+str(odor_index+1)]

        trial[odor_index]['activeChannels'] = activeChannels
        trial[odor_index]['nActiveChannels'] = nActiveChannels

        # added later by other routines and importation
        # but we should expect them to be there
        trial[odor_index]['segmentationMask'] = None

        trial[odor_index]['cells'] = None
        trial[odor_index]['normalizedCells'] = None

        trial[odor_index]['trialNumber'] = None # done when storing doc

        trial[odor_index]['difference_image'] = None  # done post alignment
 
        # extract and store image data
        
        trial[odor_index]['images'] = {}
        trial[odor_index]['images']['chan1'] = np.array([])
        trial[odor_index]['images']['chan2'] = np.array([])
        trial[odor_index]['images']['chan3'] = np.array([])
        trial[odor_index]['images']['chan4'] = np.array([])

        for chanNum in range(4):
            if activeChannels[chanNum]:
                offset = odor_index * single_odor_frame_length_with_blank
                trial[odor_index]['images']['chan'+str(chanNum+1)] = raw_image_in_channels[chanNum][:,:,offset:offset+single_odor_frame_length].copy()

    return trial # a list of single trial odor exposure dictionaries



def readMultiOdorEpoch(epochNumber, optionalParams=''):
    """
    Parse a multi odor epoch of scanimage data in the current directory.

    Takes a number and
    Arguments:
    - `epochNumber`: Integer, the epoch number to parse
    - `optionalParams`: nested dictionary that adds and/or overwrites values in the headerfile
    """

    epoch='e'+str(epochNumber)

    epochFileNames = glob.glob('*'+epoch+'*')

    tif_files = [fname for fname in epochFileNames if 'tif' in fname and not 'mean' in fname]

    # check to make sure the sizes of all tif files is the same.  Highlight anything that doesn't have the
    # mode filesize

    tif_file_sizes = np.array([os.path.getsize(tif_file) for tif_file in tif_files])
    tif_file_size_mode = int(scipy.stats.mode(tif_file_sizes)[0])

    # assume that all tifs are good to start
    goodTrials = np.logical_not(np.zeros_like(tif_files, dtype=bool))

    # check for any files that are different than the most common size, and ignore them
    file_size_same_as_mode = tif_file_sizes == tif_file_size_mode
    goodTrials = np.logical_and(goodTrials, file_size_same_as_mode)

    # check for any files that don't have an associated header file
    associated_header_file = np.array([os.path.exists(tif_name.replace('.tif', '_hdr.txt')) for tif_name in tif_files])
    goodTrials = np.logical_and(goodTrials, associated_header_file)

    tif_files = [tif[1] for tif in zip(goodTrials, tif_files) if tif[0]]
    txt_files = [tif_name.replace('.tif', '_hdr.txt') for tif_name in tif_files]
    numTrials=len(tif_files)

    print 'Reading the the following files: %s' % tif_files
    print 'With the following txt headers: %s' % txt_files

    # read in files into a list of epoch dictionaries.
    epochs=addTrialToEpoch([], tif_files[0], txt_files[0], optionalParams)
    for trial in range(1, numTrials):
        epochs=addTrialToEpoch(epochs, tif_files[trial], txt_files[trial], optionalParams)

    epochs=[DotDict(e) for e in epochs]
    return epochs

def addTrialToEpoch(epochs, tif_fn, header_fn, optionalParams):
    """
    Adds a trial's worth of data to the epoch structure 'epoch'

    If epochs is empty, it creates the basic structure, otherwise
    the new trial is cut according to odor and appended into the structure

    see: *** for structure details

    Arguments:
    - `epochs`: nil, or a list of epoch structures (nested dictionaries)
    - `tif_fn`: String, name of a tif file in the current dir to read
    - `header_fn`: String, name of the header txt file
    - `optionalParams`: nested dictionary that adds and/or overwrites values in the headerfile
    """

    # read in header file and create nested state dictionary

    state={}
    state = parseSIHeaderFile(header_fn)

    # parse optional Parameters
    optionalParams=optionalParams.split(';')
    for line in optionalParams:
        key,sep,value = line.partition('=')
        keyList = key.split('.')
        value = value.strip('\'\"')
        state=addLineToStateVar(keyList[1:],value,state)


    state['olfactometer']['odorStateList'] = state['olfactometer']['odorStateListString'].split(';')
    state['olfactometer']['odorTimeList']  = state['olfactometer']['odorTimeListString'].split(';')
    state['olfactometer']['odorFrameList'] = state['olfactometer']['odorFrameListString'].split(';')

    # calc base,odor,post for all epochs

    pre   = int(state['olfactometer']['odorFrameList'][0])
    odor  = int(state['olfactometer']['odorFrameList'][1])
    post  = int(state['olfactometer']['odorFrameList'][2])
    blank = int(state['olfactometer']['odorFrameList'][3])

    # make a list of AD channels acquired
    # set makes items unique, the list comp here gives us
    # the first 3 chars of every file in the directory
    AD_list = [i for i in set([f[0:3] for f in glob.glob('AD*.mat')])]

    # make a list of imaging channels acquired

    activeChannels = map(int, [state['acq']['savingChannel1'],
                               state['acq']['savingChannel2'],
                               state['acq']['savingChannel3'],
                               state['acq']['savingChannel4']])

    nActiveChannels=sum(activeChannels)

    nOdors = len([odorState for odorState in state['olfactometer']['odorStateList'] if odorState != '0'])

    basename = tif_fn[0:-4].split('_')[0]
    epoch = tif_fn[0:-4].split('_')[1][1:]
    acqNum = tif_fn[0:-4].split('_')[2]

    # If this is the first trial, then build up some meta data from state
    # *** this should be factored out
    if (not epochs):
        newEpochArray  = [{} for i in range(nOdors)]
        odor1Names     = {}
        odor2Names     = {}
        odor1Dilutions = {}
        odor2Dilutions = {}
        odorIndex      = 0

        for valveNumber in map(str,range(1,16)):
            if (int(state['olfactometer']['valveEnable_'+valveNumber])):
                odor1Names[odorIndex]     = state['olfactometer']['valveOdor1Name_'+valveNumber]
                odor2Names[odorIndex]     = state['olfactometer']['valveOdor2Name_'+valveNumber]
                odor1Dilutions[odorIndex] = state['olfactometer']['valveOdor1Dilution_'+valveNumber]
                odor2Dilutions[odorIndex] = state['olfactometer']['valveOdor2Dilution_'+valveNumber]
                odorIndex+=1

        for i in range(nOdors):
            # basic metadata from header
            try:
                newEpochArray[i]['baseName'] = state['files']['baseName'][0:-1]
            except KeyError:
                newEpochArray[i]['baseName'] = basename                

            try:
                newEpochArray[i]['epochNumber'] = state['epoch']
            except KeyError:
                newEpochArray[i]['epochNumber'] = epoch

            newEpochArray[i]['date']           = state['internal']['startupTimeString']
            newEpochArray[i]['resolution']     = (state['acq']['linesPerFrame'] + 'x' +
                                                  state['acq']['linesPerFrame'] + 'x' +
                                                  state['acq']['msPerLine'] + 'ms')
            newEpochArray[i]['baselineFrames'] = [0, pre]
            newEpochArray[i]['odorFrames']     = [pre, pre+odor]
            newEpochArray[i]['postFrames']     = [pre+odor+1, pre+odor+1+post]
            newEpochArray[i]['odor1Name']      = odor1Names[i]
            newEpochArray[i]['odor2Name']      = odor2Names[i]
            newEpochArray[i]['odor1Dilution']  = odor1Dilutions[i]
            newEpochArray[i]['odor2Dilution']  = odor2Dilutions[i]

            # position data
            newEpochArray[i]['position'] = state['motor']

            # AD / phys fields
            newEpochArray[i]['AD']={};
            for AD in AD_list:
                newEpochArray[i]['AD'][AD]={};
                newEpochArray[i]['AD'][AD]['allTraces'] = []
                newEpochArray[i]['AD'][AD]['meanTrace'] = []


            newEpochArray[i]['nAD'] = len(AD_list)

            newEpochArray[i]['images']={}
            # Imaging fields
            newEpochArray[i]['images']['chan1'] = np.array([], dtype='uint16')
            newEpochArray[i]['images']['chan2'] = np.array([], dtype='uint16')
            newEpochArray[i]['images']['chan3'] = np.array([], dtype='uint16')
            newEpochArray[i]['images']['chan4'] = np.array([], dtype='uint16')

            newEpochArray[i]['averageTrials'] = {}
            newEpochArray[i]['averageTrials']['chan1'] = np.array([], dtype='uint16')
            newEpochArray[i]['averageTrials']['chan2'] = np.array([], dtype='uint16')
            newEpochArray[i]['averageTrials']['chan3'] = np.array([], dtype='uint16')
            newEpochArray[i]['averageTrials']['chan4'] = np.array([], dtype='uint16')

            newEpochArray[i]['activeChannels']  = activeChannels
            newEpochArray[i]['nActiveChannels'] = nActiveChannels

            newEpochArray[i]['trialsInAverage'] = []

            newEpochArray[i]['nTrials'] = 0
            newEpochArray[i]['AD_list'] = AD_list

            newEpochArray[i]['fn_data'] = ''
            newEpochArray[i]['fn_tifs'] = []

            # all misc header info
            newEpochArray[i]['allHeaderData'] = state
    else:
        newEpochArray = epochs


    # bump the number of trials and by default include the new trial in the
    # average
    for odorEpoch in newEpochArray:
        odorEpoch['nTrials']+=1;
        odorEpoch['trialsInAverage'].append(True)

    # read data
    # image data
    print '\n'
    print '--------------------------------------------------'
    print 'reading raw image file: %s ... ' % tif_fn,
    sys.stdout.flush()
    image=imaging.io.imread(tif_fn)
    nFrames = image.shape[2]
    print 'done.'
    print '--------------------------------------------------'

    
    # *** need to change to call XSG instead.
    AD_files=glob.glob('*AD*_%s.mat' % acqNum)
    AD_waves={}
    for file in AD_files:
        wavename=file.split('.')[0]
        matfile = scipy.io.loadmat(file)
        matfile_data=matfile[wavename][0][0][0].T      # transpose is cruical for speed, alternatively, .squeeze

        AD_waves[wavename[0:3]]=matfile_data

    #split all the datas

    # first, let's demultiplex image into it's channels
    imageSize=image.shape

    imageChannels=[np.zeros([imageSize[0], imageSize[0], imageSize[2]/nActiveChannels])]*4

    for chanNum in range(4):
        if (activeChannels[chanNum]):
            imageChannels[chanNum] = np.array(image[:,:,chanNum:nFrames:nActiveChannels])
        else:
            imageChannels[chanNum]=[]

    # next, define variables for specifying limits:  start and deltas for frames and time points
    frameLimits, timeLimits=np.zeros((2,nOdors))

    state['olfactometer']['odorFrameList']=map(int,state['olfactometer']['odorFrameList'])
    state['olfactometer']['odorTimeList']=map(float,state['olfactometer']['odorTimeList'])

    frameBase = 0
    timeBase = 0

    frameDelta=state['olfactometer']['odorFrameList'][0]+state['olfactometer']['odorFrameList'][1]+state['olfactometer']['odorFrameList'][2]
    timeDelta=state['olfactometer']['odorTimeList'][0]+state['olfactometer']['odorTimeList'][1]+state['olfactometer']['odorTimeList'][2]

    # make ordered list of odor presentations
    # note indicies are zero order, but valves are 1 order
    orderedOdors = [int(odorState) for odorState in state['olfactometer']['odorStateList'] if odorState != '0']
    minValve=min(orderedOdors)
    orderedOdors = [i-minValve for i in orderedOdors]

    # and finally loop over odors, breaking image and AD data apart for each odor

    for index, odor in enumerate(orderedOdors):

        frameLimits=[frameBase, frameBase+frameDelta];
        timeLimits=[timeBase, timeBase+timeDelta];

        print('--------------------------------------------------');
        print('odor #%i (frames %i to %i, times %i to %i) is %s' % (index+1, frameLimits[0], frameLimits[1], timeLimits[0], timeLimits[1], newEpochArray[odor]['odor1Name']))
        print('--------------------------------------------------');

        for channelNumber, channelFlag in enumerate(activeChannels):
            if channelFlag:
                if not epochs:
                    newImageArray = np.expand_dims(imageChannels[channelNumber][:,:,frameLimits[0]:frameLimits[1]],axis=3)
                    if (newImageArray.any()):
                        newEpochArray[odor]['images']['chan'+str(channelNumber+1)] = newImageArray
                    else:
                        print 'in addTrailToEpoch: odor is null, adding all ones!'
                        newEpochArray[odor]['images']['chan'+str(channelNumber+1)] = np.ones((imageSize[0],imageSize[1],frameDelta,1))
                else:
                    newImageArray=np.expand_dims(imageChannels[channelNumber][:,:,frameLimits[0]:frameLimits[1]], axis=3)
                    if (newImageArray.any()):
                        newEpochArray[odor]['images']['chan'+str(channelNumber+1)]=np.concatenate((newEpochArray[odor]['images']['chan'+str(channelNumber+1)],
                                                                                                   newImageArray), axis=3)
                    else:
                        print 'in addTrailToEpoch: odor is null, adding all ones!'
                        newEpochArray[odor]['images']['chan'+str(channelNumber+1)]=np.concatenate((newEpochArray[odor]['images']['chan'+str(channelNumber+1)],
                                                                                                   np.ones((imageSize[0],imageSize[1],frameDelta,1))), axis=3)


        for AD_name in AD_list:
            if not epochs:
                newEpochArray[odor]['AD'][AD_name]['allTraces'] = np.atleast_2d(AD_waves[AD_name][timeLimits[0]*10:timeLimits[1]*10])
            else:
                try:
                    newEpochArray[odor]['AD'][AD_name]['allTraces'] = np.concatenate(( newEpochArray[odor]['AD'][AD_name]['allTraces'],
                                                                                      AD_waves[AD_name][timeLimits[0]*10:timeLimits[1]*10] ),axis=1)
                except ValueError:
                    print "ERROR IN IMPORTING AD WAVE, not enough points?"


        frameBase = frameBase + frameDelta + state['olfactometer']['odorFrameList'][3]
        timeBase = timeBase + timeDelta + state['olfactometer']['odorTimeList'][3]

    return newEpochArray


#### extractions and calculations

def calculateEpochAverages(epoch,averageImagesFlag,averageADFlag):
    """

    Arguments:
    - `epoch`: epoch dict structure
    - `averageImagesFlag`: True/False
    - `averageADFlag`: True/False
    """

    if averageImagesFlag:
        for j in range(len(epoch['activeChannels'])):
            if epoch['activeChannels'][j]:
                try:
                    epoch['averageTrials']['chan'+str(j+1)]=np.average(epoch['images']['chan'+str(j+1)][:,:,:,epoch['trialsInAverage']],axis=3).astype('uint16');
                except ValueError:
                    print "Error in calcuating average image in channel %d, possible inconsistancies across trials" % j+1
                except IndexError:
                    print 'Error in calculateTrialAverages: Index error: only one trial?'
                    epoch['averageTrials']['chan'+str(j+1)]=epoch['images']['chan'+str(j+1)][:,:,:,0].astype('uint16')

    if averageADFlag:
        for currentAD in epoch['AD_list']:
            epoch['AD'][currentAD]['meanTrace'] = np.average(epoch['AD'][currentAD]['allTraces'][:,epoch['trialsInAverage']], axis=1);

    return epoch



def buildCellDataFromEpoch(epoch,baselineFrames=[1,15], odorFrames=[29,30], subAverage=False):
    epoch['cells'] = {}
    epoch['cells']['allTraces'] = imaging.segmentation.extractTimeCoursesFromStack(epoch['images']['chan1'], epoch['segmentationMask'])
    epoch['cells'] = averageCellData(epoch['cells'])

    # calcuate baselined dF/F
    epoch['normalizedCells'] = {}
    epoch['normalizedCells']['allTraces'] = epoch['cells']['allTraces'].copy()

    if subAverage:
        #        averageSignal = np.expand_dims(epoch['normalizedCells']['allTraces'][:,0,:].copy(), 1)
        #        epoch['normalizedCells']['allTraces'] -= averageSignal

        # calculate local average for subtraction
        for ROI in range(1, epoch['normalizedCells']['allTraces'].shape[1]):
            mask_dilated = epoch['segmentationMask'] == ROI
            for i in range(3):
                mask_dilated = pymorph.dilate(mask_dilated)

            mask_dilated = np.logical_and(mask_dilated, np.logical_not(pymorph.dilate(epoch['segmentationMask']==ROI)))
            
            local_neuropil_mask = np.logical_and(mask_dilated, epoch['segmentationMask'] == 0)
            local_neuropil_signals = imaging.segmentation.avgFromROIInStack(epoch['images']['chan1'], local_neuropil_mask)
            local_neuropil_signals = local_neuropil_signals.astype(float) - np.atleast_2d(np.mean(local_neuropil_signals[baselineFrames[0]:baselineFrames[1], :], axis=0))

            for trial in range(epoch['nTrials']):
                local_neuropil_signals[:, trial] = traces.boxcar(local_neuropil_signals[:,trial])
                #            plt.matshow(local_neuropil_signals)

            epoch['normalizedCells']['allTraces'][:,ROI,:] -= local_neuropil_signals

    epoch['normalizedCells'] = normalizeCellData(epoch['normalizedCells'],baselineFrames)
    epoch['normalizedCells'] = baselineCellData(epoch['normalizedCells'],baselineFrames)

    # if subAverage:
    #     #        averageSignal = np.expand_dims(epoch['normalizedCells']['allTraces'][:,0,:].copy(), 1)
    #     #        epoch['normalizedCells']['allTraces'] -= averageSignal

    #     # calculate local average for subtraction
    #     for ROI in range(1, epoch['normalizedCells']['allTraces'].shape[1]):
    #         mask_dilated = epoch['segmentationMask'] == ROI
    #         for i in range(5):
    #             mask_dilated = pymorph.dilate(mask_dilated)
    #         local_neuropil_mask = np.logical_and(mask_dilated, epoch['segmentationMask'] == 0)
    #         local_neuropil_signals = imaging.segmentation.avgFromROIInStack(epoch['images']['chan1'], local_neuropil_mask)

    #         local_neuropil_signals = local_neuropil_signals.astype(float) / np.atleast_2d(np.mean(local_neuropil_signals[baselineFrames[0]:baselineFrames[1], :], axis=0))
    #         local_neuropil_signals -= 1.0
    #         for trial in range(epoch['nTrials']):
    #             local_neuropil_signals[:, trial] = traces.boxcar(local_neuropil_signals[:,trial])

    #         epoch['normalizedCells']['allTraces'][:,ROI,:] -= local_neuropil_signals

    epoch['normalizedCells'] = averageCellData(epoch['normalizedCells'])
    epoch['normalizedCells'] = calculateCellDataAmplitudes(epoch['normalizedCells'], baselineFrames, odorFrames)

    return DotDict(epoch)

def calculateCellDataAmplitudes(cellData, baselineFrames, stimFrames, fieldOfViewPositionOffset=[0,0,0], stdThreshold=2):
    nCells,nTimePoints,nTrials = cellData['allTraces'].shape

    cellData['baselineAverage'] = np.mean(cellData['allTraces'][baselineFrames[0]:baselineFrames[1],:,:],axis=0)
    cellData['baselineSTD'] = scipy.std(cellData['allTraces'][baselineFrames[0]:baselineFrames[1],:,:],axis=0)

    cellData['stimAverage'] = np.mean(cellData['allTraces'][stimFrames[0]:stimFrames[1],:,:],axis=0)
    cellData['deltaValues'] = np.squeeze(np.array([t[1]-t[0] for t in zip(cellData['baselineAverage'],cellData['stimAverage'])]))

    # note this works because the deltas are post-baselining
    cellData['responders']= cellData['deltaValues'] > stdThreshold * cellData['baselineSTD'] # list of True for each cell by default

    cellData['position']=[fieldOfViewPositionOffset for i in range(cellData['allTraces'].shape[0])] # list of 0,0,0 for each cell by default

    cellData['meanDelta'] = np.mean(cellData['deltaValues'],axis=1)
    cellData['meanBaseline'] = np.mean(cellData['baselineValues'],axis=1)
    cellData['meanResponders'] = np.mean(cellData['responders'],axis=1)>=0.25
    cellData['meanResponders'][0] = 0
    return cellData

def calculateResponderMasks(epoch):
    objectLabels = [i for i in set(epoch['segmentationMask'].ravel()) if i>0]
    nTrials = epoch['normalizedCells']['allTraces'].shape[2]
    epoch['responderMasks']=np.zeros((epoch['segmentationMask'].shape[0], epoch['segmentationMask'].shape[1], nTrials))
    epoch['meanResponderMask']=np.zeros((epoch['segmentationMask'].shape[0], epoch['segmentationMask'].shape[1]))

    for trial in range(nTrials):
        epoch['responderMasks'][:,:,trial] = epoch['segmentationMask'].copy(order='A')

        for cellIndex, cell in enumerate(objectLabels):
            index = epoch['segmentationMask'] == cell
            epoch['responderMasks'][:,:,trial][index] = epoch['normalizedCells']['responders'][cellIndex,trial]

    epoch['meanResponderMask'] = epoch['segmentationMask'].copy(order='A')
    for cellIndex, cell in enumerate(objectLabels):
        index = epoch['segmentationMask'] == cell
        epoch['meanResponderMask'][index] = epoch['normalizedCells']['meanResponders'][cellIndex]
    return epoch

def baselineCellData(cellData, baselineFrames):
    cellData['baselineValues'] = np.expand_dims(np.mean(cellData['allTraces'][baselineFrames[0]:baselineFrames[1],:,:],axis=0),0)
    cellData['allTraces'] = cellData['allTraces'] - cellData['baselineValues']
    return cellData

def normalizeCellData(cellData, normFrames):
    cellData['normValues'] = np.expand_dims(np.mean(cellData['allTraces'][normFrames[0]:normFrames[1],:,:],axis=0),0)
    cellData['allTraces'] = cellData['allTraces'].astype('float') / cellData['normValues']
    return cellData

def averageCellData(cellData):
    # allTraces is time x cell x trial
    cellData['meanTraces'] = np.mean(cellData['allTraces'], axis=2) 
    cellData['sem'] = scipy.stats.sem(cellData['allTraces'], axis=2)
    return cellData

#### plotting

def plotNormCells(epoch):
    f=plt.figure(num=epoch.odor1Name+r'_'+str(np.random.randint(1000000)))
    a=f.add_subplot(111)
    for index in range(epoch.normalizedCells.allTraces.shape[0]):
        a.plot(epoch.normalizedCells.allTraces[index,:,:])

def plotMeanCells(epoch):
    f=plt.figure(num=epoch.odor1Name+'_'+str(np.random.randint(1000000)))
    a=f.add_subplot(111)
    for index in range(epoch.normalizedCells.meanTraces.shape[0]):
        a.plot(epoch.normalizedCells.meanTraces[index,:])
        plt.ylim(-.05,.15)

def plotRespondersAndNonResponders(epoch):
    f=plt.figure(num=epoch.odor1Name+'_'+str(np.random.randint(1000000)))
    a=f.add_subplot(211)
    b=f.add_subplot(212)
    for cell in range(epoch.normalizedCells.allTraces.shape[0]):
        for trial in range(epoch.normalizedCells.allTraces.shape[2]):
            if epoch.normalizedCells.responders[cell,trial]:
                a.plot(epoch.normalizedCells.allTraces[cell,:,trial])
            else:
                b.plot(epoch.normalizedCells.allTraces[cell,:,trial])

    # plt.subplot(211)
    # plt.ylim(-.1,.25)
    # plt.subplot(212)
    # plt.ylim(-.1,.25)

def plotMeanRespondersAndNonResponders(epoch):
    f=plt.figure(num=epoch.odor1Name+'_means'+'_'+str(np.random.randint(1000000)))
    a=f.add_subplot(211)
    b=f.add_subplot(212)
    for cell in range(epoch.normalizedCells.allTraces.shape[0]):
        if epoch.normalizedCells.meanResponders[cell]:
            a.plot(epoch.normalizedCells.meanTraces[cell,:])
        else:
            b.plot(epoch.normalizedCells.meanTraces[cell,:])

    # plt.subplot(211)
    # plt.ylim(-.1,.12)
    # plt.subplot(212)
    # plt.ylim(-.1,.12)


def parseXSG(filename):
    xsg = scipy.io.matlab.loadmat(filename)

    # fucking absurd nested datafile
    # default sample rate is 10kHz

    # need to actually loop over the number of channels...
    
    # for a single channel:
    # actual array

    d={}
    chanName = xsg['data'][0][0][1][0][0][2][0]
    
    d[chanName]['data']       = np.squeeze(xsg['data'][0][0][1][0][0][0])
    d[chanName]['sampleRate'] = 10000

    return d
