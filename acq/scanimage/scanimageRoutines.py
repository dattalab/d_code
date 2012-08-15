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


__all__ = ['readRawSIImages', 'parseSIHeaderFile', 'importTrial',
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

    print 'Reading ' + tif_filename.split('.tif')[0] + '...'
    sys.stdout.flush()
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

    for i, odor_index in enumerate(odor_list_indicies):
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
#                offset = odor_index * single_odor_frame_length_with_blank
                offset = i * single_odor_frame_length_with_blank

#                print trial[odor_index]['odor1Name'], odor_index, offset, offset+single_odor_frame_length
                trial[odor_index]['images']['chan'+str(chanNum+1)] = raw_image_in_channels[chanNum][:,:,offset:offset+single_odor_frame_length].copy()

    return trial # a list of single trial odor exposure dictionaries


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
