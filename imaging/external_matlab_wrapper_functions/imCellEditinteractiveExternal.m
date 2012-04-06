function imCellEditinteractiveExternal()

    % assumes a file 'temp.hdf5' with fovimage data

    fovimage=hdf5read('temp.hdf5','fovimage');
    fovimage=permute(fovimage, [2 1]);

    seedPoints=hdf5read('temp.hdf5','seedPoints');
    seedPoints=permute(seedPoints, [2 1]);
    
    size(fovimage)

    bwOut = imCellEditInteractive(fovimage, [], [], 3, seedPoints);
    mask = bwlabel(bwOut);

    hdf5write('temp.hdf5', 'bwOut', uint8(permute(bwOut,[2 1])),'WriteMode','append')
    hdf5write('temp.hdf5', 'mask', uint8(permute(mask,[2 1])),'WriteMode','append')

    exit
end