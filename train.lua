
require 'torch'
require 'nn'
require 'optim'
require 'lfs'
require 'gnuplot'
require 'util.print'
require 'util.misc'

local MODEL_ID = torch.randn(1)[1]
local ImageMinibatchLoader = require 'util.ImageMinibatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a cnn to classify EEG recordings')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/train','data directory')
cmd:option('-torch_dir','data/torch','torch data directory')
cmd:option('-test_dir','data/test','test data directory')
cmd:option('-apply_zca',true,'specifies if ZCA whitening should be applied to network input')
cmd:option('-val_part',0.07,'what fraction of data should be taken as validation set')
cmd:option('-patches_per_file',200,'how many patches to extract from each training file')
cmd:option('-patch_size',48,'patch size')
cmd:option('-batch_size',10,'minibatch size')
-- model prototype
cmd:option('-proto_file', 'cnn/proto/first_cnn.lua', 'file defining network structure')
-- optimization
cmd:option('-optim_algo','rmsprop','optimization algorithm')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
-- checkpoints
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-plot_after',100,'how many steps/minibatches to ignore in the plot')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','cnn','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then printRed('package cunn not found!') end
    if not ok2 then printRed('package cutorch not found!') end
    if ok and ok2 then
        printGreen('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        printYellow('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        printYellow('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        printYellow('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local loader = ImageMinibatchLoader.create(opt)

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
local start_iter = 1
if string.len(opt.init_from) > 0 then
    printRed('Checkpoints aren\'t supported yet!')
    os.exit()
else
    print('creating CNN')
    dofile(opt.proto_file)
    criterion = nn.MSECriterion()
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    cnn:cuda()
    criterion:cuda()
end

-- evaluate the loss over an entire split
function eval_split(split_index)
    print('evaluating loss over split index ' .. split_index)

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0

    -- TODO: dirty hack. will work as long as there are less then 1e6 batches in a file
    function get_batch_id()
        return loader.file_idx[split_index] * 1e6 + loader.batch_idx[split_index]
    end

    -- iterate over batches in the split
    local ct = 0
    local last_batch_id = -1
    local example_shown = false
    local examples = {}
    while get_batch_id() > last_batch_id do
        last_batch_id = get_batch_id()
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            for i = 1, #x do
                x[i] = x[i]:float():cuda()
                y[i] = y[i]:float():cuda()
            end
        end
        -- forward pass
        local partial_loss = 0
        for i = 1, #x do
            partial_loss = partial_loss + criterion:forward(cnn:forward(x[i]):clamp(0,1), y[i])
            if not example_shown then
                table.insert(examples, x[i])
                table.insert(examples, cnn.output:clone())
            end
        end
        if not example_shown then
            local patch_size = examples[1]:size(2)
            local example_tensor = torch.Tensor(patch_size, patch_size * #examples)
            for i = 1, #examples do
                example_tensor:sub(1, patch_size, (i-1) * patch_size + 1, i * patch_size):copy(examples[i]:view(patch_size, -1))
            end
            gnuplot.figure(2)
            gnuplot.imagesc(example_tensor:clamp(0, 1), 'gray')
            gnuplot.figure(1)
            example_shown = true
        end

        loss = loss + (partial_loss / #x)
        ct = ct + 1
        if ct % 10 == 0 then
            print('Evaluated: ' .. ct .. ' batches')
        end
    end

    loss = loss / ct
    return loss
end

local params, grad_params = cnn:getParameters()
print('number of parameters: ' .. params:nElement())
-- params:uniform(-0.08, 0.08)
local feval = function(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        for i = 1, #x do
            x[i] = x[i]:float():cuda()
            y[i] = y[i]:float():cuda()
        end
    end

    local loss = 0
    for i = 1, #x do
        local partial_loss = criterion:forward(cnn:forward(x[i]):clamp(0, 1), y[i])
        loss = loss + partial_loss
        cnn:backward(x[i], criterion:backward(cnn.output, y[i]))
    end
    loss = loss / #x

    grad_params:clamp(-5, 5)
    grad_params:div(#x)
    return loss, grad_params
end


-- start optimization here
train_losses = train_losses or {}
train_losses_avg = train_losses_avg or {}
val_losses = val_losses or {}

local optim_fun, optim_state
if opt.optim_algo == 'rmsprop' then
    optim_fun = optim.rmsprop
    optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
elseif opt.optim_algo == 'adadelta' then
    optim_fun = optim.adadelta
    optim_state = {rho = 0.95, eps = 1e-7}
end

local iterations = opt.max_epochs * loader.total_samples
local loss0 = nil
for i = start_iter, iterations do
    local epoch = i / loader.total_samples

    local timer = torch.Timer()
    local _, loss = optim_fun(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss
    train_losses_avg[i] = calculate_avg_loss(train_losses)

    if i % opt.print_every == 0 then
        local grad_norm = grad_params:norm()
        local param_norm = params:norm()
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, param norm = %.2e time/batch = %.2fs",
                i, iterations, epoch, train_loss, grad_norm / param_norm, param_norm, time))
        if i > opt.plot_after then
            local ct = opt.plot_after;
            local xAxis = torch.Tensor(#train_losses_avg - opt.plot_after):apply(function() ct = ct + 1; return ct; end)
            gnuplot.plot(xAxis, torch.Tensor(train_losses_avg):sub(opt.plot_after + 1, i))
        end
    end

    -- exponential learning rate decay
    if i % loader.total_samples == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.4f_%.2f.t7', opt.checkpoint_dir, opt.savefile, val_loss, epoch)
        printGreen('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.cnn = cnn
        checkpoint.criterion = criterion
        checkpoint.patch_size = opt.patch_size
        checkpoint.type = "cnn"
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.loader = {}
        checkpoint.loader.file_idx = loader.file_idx
        checkpoint.loader.batch_idx = loader.batch_idx
        checkpoint.id = MODEL_ID
        torch.save(savefile, checkpoint)
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    --if loss0 == nil then loss0 = loss[1] end
    --if loss[1] > loss0 * 3 then
        --print('loss is exploding, aborting.')
        --break -- halt
    --end
end


print 'TRAINING DONE'
