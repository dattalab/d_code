# scanimage specific analysis functions

import sys
import os
import numpy as np

import copy


import scipy.ndimage
import imaging.io

from ephusRoutines import parseXSG


__all__ = ['readRawSIImages', 'parseSIHeaderFile', 'importTrial']

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

    return state


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

    single_odor_time_length_with_blank_in_seconds = single_odor_frame_length_with_blank * 1.0 / float(state['acq']['frameRate'])

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
    
    # read in the xsg file
    try:
        xsg_sub_dir_name = state['xsgFilename'].split('\\')[-2]
        xsg_filename = state['xsgFilename'].split('\\')[-1]
        xsg_file = parseXSG(os.path.join(xsg_sub_dir_name, xsg_filename))
    except IOError:
        print 'Error reading associated xsg, ' + os.path.join(xsg_sub_dir_name, xsg_filename) + ', doesn\'t exist?'
        xsg_file = None
    except:
        print 'No associated xsg file in header file'
        xsg_file = None


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

    trial = {}
    for odor_index in odor_list_indicies:
        trial[odor_index] = {}

    for i, odor_index in enumerate(odor_list_indicies):

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
        for chanNum in range(4):
            if activeChannels[chanNum]:
                offset = i * single_odor_frame_length_with_blank

                trial[odor_index]['images']['chan'+str(chanNum+1)] = raw_image_in_channels[chanNum][:,:,offset:offset+single_odor_frame_length].copy()
            else:
                trial[odor_index]['images']['chan'+str(chanNum+1)] = np.array([])

        # extract and store xsg info
        trial[odor_index]['xsg'] = copy.deepcopy(xsg_file)
        if trial[odor_index]['xsg'] is not None:
            for prog in ['ephys', 'acquirer', 'stimulator']:
                if xsg_file[prog] is not None:
                    single_odor_samples = int(single_odor_time_length_with_blank_in_seconds * 10000)
                    offset = i * single_odor_samples
                    for channel in xsg_file[prog].keys():
                        trial[odor_index]['xsg'][prog][channel] = trial[odor_index]['xsg'][prog][channel][offset:offset+single_odor_samples] 

    return trial.values() # a list of single trial odor exposure dictionaries

