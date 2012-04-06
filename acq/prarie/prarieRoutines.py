import imaging_analysis.io
from elementtree.ElementTree import parse as elementTreeParse

__all__ = ['loadImageSeriesFromXML']

def loadImageSeriesFromXML(xmlFileName):
    """
    Parse PrarieView XML file and read in associated tiff files and acquisition values
    Could be expanded.

    Returns numpy array of channel1, channel2, and a list containing imaging parameters
    dictionaries for each channel

    chan1, chan2, keys = loadImageSeriesFromXML('blah.xml')

    :param xmlFileName: name of PrarieView xml file to be parsed
    :returns: a nested tuple ((channel1_file_list, channel2_file_list), keylist)
    """

    # one for each channel
    fileList = {'channel_1':[], 'channel_2':[]}
    keyList = {'channel_1':[], 'channel_2':[]}


    p=elementTreeParse(xmlFileName)
    elem=p.getroot()
    seq=elem.getchildren()[1]
    for frame in seq.getchildren():
        frameElements = frame.getchildren()
        file = frameElements[0]
        extraParameters = frameElements[1]
        PVState = frameElements[2]

        channel = 'channel_' + file.attrib['channel']
        fileList[channel].append(file.attrib['filename'])
        
        keyDict = {}
        for keyElement in PVState.getchildren():
            keyDict[keyElement.attrib['key']] = keyElement.attrib['value']
        keyList[channel].append(keyDict)

    return imaging_analysis.io.readImagesFromList(fileList['channel_1']), imaging_analysis.io.readImagesFromList(fileList['channel_2']), keyList

