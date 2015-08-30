require 'nn'

cnn = nn.Sequential()
cnn:add( nn.SpatialConvolution(1, 20, 9, 9, 1, 1, 4, 4) )
cnn:add( nn.ReLU() )
cnn:add( nn.SpatialConvolution(20, 20, 9, 9, 1, 1, 4, 4) )
cnn:add( nn.ReLU() )
cnn:add( nn.SpatialConvolution(20, 1, 3, 3, 1, 1, 1, 1) )
cnn:add( nn.ReLU() )
