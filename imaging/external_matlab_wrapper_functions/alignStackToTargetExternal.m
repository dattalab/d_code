function alignStackToTargetExternal()

    % assumes a file 'temp.hdf5' with stack and target datasets

    stack=hdf5read('temp.hdf5','stack');
    target=hdf5read('temp.hdf5','target');
    mode=hdf5read('temp.hdf5', 'mode');
    
    target=permute(target, [2 1]);
    stack=permute(stack,[4 3 2 1]);

    alignedStack=zeros(size(stack));

    for i = 1:size(stack,4)
        alignedStack(:,:,:,i) = stackRegisterJavaAG(stack(:,:,:,i), ...
                                                    target, mode);
    end
    hdf5write('temp.hdf5', 'alignedStack', uint16(permute(alignedStack,[4 3 2 1])),'WriteMode','append')

    exit
end