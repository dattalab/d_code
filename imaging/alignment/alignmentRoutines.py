# alignment code
# most is wrappers for matlab functions--- make sure these are in your path!
# the code we rely on is based on the imagej turboreg plugin
import numpy as np
import os
import h5py
import subprocess

__all__ = ['alignSeries','alignStack']

def alignSeriesMATLAB(series, mode='translation', supressOutput=True):
    """This is a wrapper function for some MATLAB code that aligns a 3d stack of images
    to a target frame.  Based on TurboReg plugin for ImageJ and some MATLAB code
    from the Reid lab.  Depends on a simple matlab function which opens the hdf5 file
    and aligns the array, and writes it back out to another part of the hdf5 file.

    Realize that this is a wrapper for alignStack - it bumps the dims, aligns, and
    squeezes the output.

    series should be 3D, X x Y x Frames

    target is np.mean(series[:,:,1:3])

    .. warning:: DEPRECATED, use alignSeries instead
    
    :param series: 3d numpy array to align on a frame by frame basis
    :param mode: one of 'translation', 'scaledRotation', 'rigidBody', 'affine'
    :param supressOutout: boolean, defaults to True
    :returns: an aligned 3d numpy array
    """

    return np.squeeze(alignStack(np.expand_dims(series, axis=3), mode, supressOutput))
    
def alignStackMATLAB(stack, mode='translation', supressOutput=True):
    """This is a wrapper function for some MATLAB code that aligns a 3d stack of images
    to a target frame.  Based on TurboReg plugin for ImageJ and some MATLAB code
    from the Reid lab.  Depends on a simple matlab function which opens the hdf5 file
    and aligns the array, and writes it back out to another part of the hdf5 file

    stack should be 4D, X x Y x Frames x Trials

    target is np.mean(stack[:,:,1:3:1])

    .. warning:: DEPRECATED, use alignStack instead
  
    :param stack: 3d or 4d numpy array to align on a frame by frame basis
    :param mode: one of 'translation', 'scaledRotation', 'rigidBody', 'affine'
    :param supressOutout: boolean, defaults to True
    :returns: an aligned 4d numpy array
    """

    modeDict = {'translation':0, 'scaledRotation':1, 'rigidBody':2, 'affine':3}

    target=np.squeeze(np.mean(stack[:,:,1:3,0],axis=2))
    
    if os.path.isfile('temp.hdf5'):
        os.system('rm -rf *.hdf5')
    f=h5py.File('temp.hdf5')
    f.create_dataset('stack',data=stack, dtype='uint16')
    f.create_dataset('target',data=target, dtype='uint16')
    f.create_dataset('mode', data=modeDict[mode])
    f.close()

    # call align code (external matlab function, which is itself a wrapper for java... yuck)
    # all series are aligned to the first part of the first series

    if supressOutput:
        handle=subprocess.Popen('matlab -nodisplay -r \'alignStackToTargetExternal\'',
                                 stdout=open('temp.txt','a+'), stdin=open('/dev/null'), shell=True, executable="/bin/bash")
    else:
        print 'Launching MATLAB to align image...\n'
        handle=subprocess.Popen('matlab -nodisplay -r \'alignStackToTargetExternal\'',
                                 stdin=open('/dev/null'), shell=True, executable="/bin/bash")
    handle.wait()

    # import the aligned stack and delete temporary files

    f=h5py.File('temp.hdf5','r')
    alignedImage=np.array(f.get('alignedStack')[:], dtype='uint16')
    f.close()

    os.system("rm -rf *.hdf5 temp.txt temperr.txt")

    return alignedImage


def alignSeries(series, mode='translation', supressOutput=True):
    """This is a wrapper function for some MATLAB code that aligns a 3d stack of images
    to a target frame.  Based on TurboReg plugin for ImageJ and some MATLAB code
    from the Reid lab.  Depends on a simple matlab function which opens the hdf5 file
    and aligns the array, and writes it back out to another part of the hdf5 file.

    Realize that this is a wrapper for alignStack - it bumps the dims, aligns, and
    squeezes the output.

    series should be 3D, X x Y x Frames

    target is np.mean(series[:,:,1:3])
    
    :param series: 3d numpy array to align on a frame by frame basis
    :param mode: one of 'translation', 'scaledRotation', 'rigidBody', 'affine'
    :param supressOutout: boolean, defaults to True
    :returns: an aligned 3d numpy array
    """

    return np.squeeze(alignStack(np.expand_dims(series, axis=3), mode, supressOutput))
    
def alignStack(stack, mode='translation', supressOutput=True):
    """This is a wrapper function for some MATLAB code that aligns a 3d stack of images
    to a target frame.  Based on TurboReg plugin for ImageJ and some MATLAB code
    from the Reid lab.  Depends on a simple matlab function which opens the hdf5 file
    and aligns the array, and writes it back out to another part of the hdf5 file

    stack should be 4D, X x Y x Frames x Trials

    target is np.mean(stack[:,:,1:3:1])

    :param stack: 3d or 4d numpy array to align on a frame by frame basis
    :param mode: one of 'translation', 'scaledRotation', 'rigidBody', 'affine'
    :param supressOutout: boolean, defaults to True
    :returns: an aligned 4d numpy array
    """

    modeDict = {'translation':0, 'scaledRotation':1, 'rigidBody':2, 'affine':3}

    target=np.squeeze(np.mean(stack[:,:,1:3,0],axis=2))

    external_java_dir = os.path.join(os.path.expandvars('$HOME'), 'Dropbox/python_modules/dattacode/imaging/external_java_scripts')

    if os.path.isfile(os.path.join(external_java_dir, 'temp.hdf5')):
        handle=subprocess.Popen('rm -rf *.hdf5 *.h5 temp.txt temperr.txt',
                                cwd=external_java_dir, shell=True, executable="/bin/bash")
    f=h5py.File(os.path.join(external_java_dir, 'temp.hdf5'))
    f.create_dataset('stack',data=stack, dtype='single')
    f.create_dataset('target',data=target, dtype='single')
    f.create_dataset('dims',data=np.array(stack.shape))
    f.create_dataset('mode', data=modeDict[mode])
    f.close()

    # call align code (turboreg imagej plugin, via java)
    # all series are aligned to the first part of the first series

    if not supressOutput:
        print 'Launching JAVA to align image...\n'
    
    handle=subprocess.Popen('java -cp .:* AlignWrapper temp.hdf5',
                            cwd=external_java_dir,
                            stdout=open('temp.txt','a+'), stdin=open('/dev/null'), shell=True, executable="/bin/bash")
    handle.wait()

    # import the aligned stack and delete temporary files

    f=h5py.File(os.path.join(external_java_dir, 'temp_out.h5'),'r')
    alignedImage=np.array(f.get('alignedStack')[:], dtype='single')
    f.close()

    # set all edges to 0 to deal with alignment artifacts
    alignedImage = alignedImage.astype('uint16')
    # threshold to deal with alignment round-off artifacts
    alignedImage[alignedImage>65250] = 0

    handle=subprocess.Popen('rm -rf *.hdf5 *.h5 temp.txt temperr.txt',
                            cwd=external_java_dir,
                            shell=True, executable="/bin/bash")
    
    return alignedImage
