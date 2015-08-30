require 'nn'

cnn = nn.Sequential()
cnn:add( nn.SpatialConvolution(1, 64, 9, 9, 1, 1, 4, 4) )
cnn:add( nn.ReLU() )
cnn:add( nn.Dropout(0.3) )
cnn:add( nn.SpatialConvolution(64, 64, 9, 9, 1, 1, 4, 4) )
cnn:add( nn.ReLU() )
cnn:add( nn.Dropout(0.3) )
cnn:add( nn.SpatialConvolution(64, 1, 3, 3, 1, 1, 1, 1) )
cnn:add( nn.ReLU() )
