function alignSeriesToTargetExternal(mode)
    % assumes a file 'temp.hdf5' with stack and target datasets

    stack=hdf5read('temp.hdf5','stack');
    target=hdf5read('temp.hdf5','target');
    mode=hdf5read('temp.hdf5', 'mode');
    
    target=permute(target, [2 1]);
    stack=permute(stack,[3 2 1]);
    
    alignedStack = stackRegisterJavaAG(stack, target, mode);
    
    hdf5write('temp.hdf5', 'alignedStack', permute(alignedStack,[3 2 1]),'WriteMode','append')

    exit

end