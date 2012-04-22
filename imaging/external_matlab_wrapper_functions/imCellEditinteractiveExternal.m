function imCellEditinteractiveExternal()

    % assumes a file 'temp.hdf5' with fovimage data

    fovimage=hdf5read('./temp.hdf5','fovimage');
    fovimage=permute(fovimage, [2 1]);

    file_info = h5info('./temp.hdf5');
    s = size(file_info.Datasets);
    num_datasets = s(1);
    
    disp(num_datasets)
    
    if num_datasets == 2
        seedPoints=hdf5read('./temp.hdf5','seedPoints');
        seedPoints=permute(seedPoints, [2 1]);
    else
        seedPoints = [];
    end
    
    size(fovimage)

    bwOut = imCellEditInteractive(fovimage, [], [], 3, seedPoints);
    mask = bwlabel(bwOut);

    hdf5write('./temp.hdf5', 'bwOut', uint8(permute(bwOut,[2 1])),'WriteMode','append')
    hdf5write('./temp.hdf5', 'mask', uint8(permute(mask,[2 1])),'WriteMode','append')

    exit
end