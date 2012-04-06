// for AlignWrapper

import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import ch.systemsx.cisd.hdf5.IHDF5Writer;
import ch.systemsx.cisd.base.mdarray.*;

import org.apache.commons.lang.ArrayUtils;

import java.lang.Math.*;

import java.io.*;
import java.awt.image.*;
import ij.*;
import ij.io.*;
import ij.gui.*;
import ij.process.*;
import ij.plugin.PlugIn;
import ij.measure.*;

public class AlignWrapper {
    public static void main(String[] args) {
	System.out.print("Loading data from " + args[0] + "...");

	// get filename from args
	String filename = args[0];

	// open file and import data
	/* PYTHON CODE EXPORTING DATA
	  f.create_dataset('stack',data=stack, dtype='uint16')
	  f.create_dataset('target',data=target, dtype='uint16')
	  f.create_dataset('mode', data=modeDict[mode])
	*/
	IHDF5Reader reader = HDF5Factory.openForReading(filename);
	MDFloatArray stack = reader.readFloatMDArray("stack");
	MDFloatArray target = reader.readFloatMDArray("target");
	int[] dims = reader.readIntArray("dims");
	int mode = reader.readInt("mode");
	reader.close();
	System.out.println(" done.");
	
	MDFloatArray alignedStack = new MDFloatArray(dims);

	System.out.println("stack dims:"+ ArrayUtils.toString(stack.dimensions()));
	
	int nRows = dims[0];
	int nCols = dims[1];
	int nFrames = dims[2];

	int nTrials=1;
	if (dims.length == 4) {
	    nTrials = dims[3];
	}

	int tempIndex;

	// parse mode, define landmarks and call alignment

	int rowOffset = (int)Math.round(nRows * 0.20);
	int colOffset = (int)Math.round(nCols * 0.20);

	String cropping = String.format("%d %d %d %d", 5, 5, nCols-6, nRows-6);

	String landmarks_1 = String.format("%d %d %d %d", (int)Math.round(nCols/2), (int)Math.round(nRows/2), (int)Math.round(nCols/2), (int)Math.round(nRows/2));
	
	//	String landmarks_2 = num2str(round([nCols nRows nCols nRows nCols nRows nCols nRows]/2) + [-colOffset 0 -colOffset 0 colOffset 0 colOffset 0]);
	//String landmarks_3 = num2str(round([nCols nRows nCols nRows nCols nRows nCols nRows nCols nRows nCols nRows]/2) + ...
	//[-colOffset -rowOffset -colOffset -rowOffset -colOffset +rowOffset -colOffset +rowOffset +colOffset -rowOffset +colOffset -rowOffset]);

	// default case of mode = 0 
	String modeString = "translation";
	String landmarks = landmarks_1;
	String landmarks_3 = landmarks_1;
	String landmarks_2 = landmarks_1;

	switch (mode) {
	case 1:
	    landmarks = landmarks_2;
	    modeString = "scaledRotation";
	case 2:
	    landmarks = landmarks_3;
	    modeString = "rigidBody";
	case 3:
	    landmarks = landmarks_3;
	    modeString = "affine";
	}
	
	String cmdstr = String.format("-align -window s %s -window t %s -%s %s -hideOutput", cropping, cropping, modeString, landmarks);
	// cmdstr = sprintf(['-align -window s %s -window t %s ' '-' mode ' %s -hideOutput'],             cropping, cropping, landmarks);

	// convert target data into an ImagePlus Object
	ImageStack ijTarget = new ImageStack(nRows, nCols);
	float[][] targetTempFrame = new float[nRows][nCols];
	for (int x=0; x<nRows; x++) {
	    for (int y=0; y<nCols; y++) {
		targetTempFrame[x][y] = target.get(x,y);
	    }
	}
	ImageProcessor sp = new FloatProcessor(targetTempFrame);
	ijTarget.addSlice("", sp);
	ImagePlus ijPlusTarget = new ImagePlus("t", ijTarget);

	IJAlign_AK ijAligner = new IJAlign_AK();
	
	// loop over every trial in the stack, create an ImagePlus object 
	for (int trial=1; trial<=nTrials; trial++) {
	    // create ijPlusStack
	    ImageStack ijStack = new ImageStack(nRows, nCols);

	    // make image	    
	    float[][] tempFrame = new float[nRows][nCols];
 	    for (int frame=0; frame<nFrames; frame++) {
		for (int x=0; x<nRows; x++) {
		    for (int y=0; y<nCols; y++) {
			tempFrame[x][y] = stack.get(x ,y, frame, trial-1);
		    }
		}
		ijStack.addSlice("", new FloatProcessor(tempFrame));
	    }

	    ImagePlus ijPlusStack = new ImagePlus("s", ijStack);

	    // call alignment
	    System.out.print("Calling alignment...");
 	    ImagePlus resultPlusStack = ijAligner.doAlign(cmdstr, ijPlusStack, ijPlusTarget);
	    System.out.println(" done.");

	    // put content of resultStack back into MDFloatArray

	    System.out.print("Saving aligned stack...");

	    resultPlusStack.setSlice(1);

	    for (int frame=0; frame<nFrames; frame++) {
		resultPlusStack.setSlice(frame+1);
		ImageProcessor resultProcessor = resultPlusStack.getProcessor();
		//		System.out.println("new: "+resultProcessor.getPixelValue(3,3));

		ijPlusStack.setSlice(frame+1);
		ImageProcessor ijProcessor = ijPlusStack.getProcessor();
		//		System.out.println("orig: "+ijProcessor.getPixelValue(3,3));

		for (int y=0; y<nCols; y++) {
		    for (int x=0; x<nRows; x++) {
			alignedStack.set(resultProcessor.getPixelValue(x,y), x, y, frame, trial-1);
		    }
		}
	    }

	}
	IHDF5Writer writer = HDF5Factory.open("temp_out.h5");
	writer.writeFloatMDArray("alignedStack", alignedStack);
	writer.close();
	System.out.println(" done.");
    }
}
