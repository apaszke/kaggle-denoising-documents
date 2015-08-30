require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'xlua'
require 'image'

require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-data_dir','data/test','directory containing sampled files')
cmd:option('-out_dir','tmp/sampled_files','directory containing sampled files')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- clear old sampled files
os.execute('rm -rf ' .. opt.out_dir)
lfs.mkdir(opt.out_dir)

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
cnn = checkpoint.cnn
patch_size = checkpoint.patch_size or 32

-- start sampling

out_file = io.open('tmp/submission.csv', 'w')
out_file:write('id,value\n')
test_img = torch.load('data/torch/data_test1.t7')
for i = 1, #test_img do
    xlua.progress(i, #test_img)

    -- load the data
    local img = test_img[i].img_data
    local img_number = test_img[i].img_number
    local height = img:size(2)
    local width = img:size(3)
    img = img:view(height, width)

    local out_img = torch.Tensor(height, width)

    local patch = torch.Tensor(patch_size, patch_size)
    for row = 1, math.ceil(height / patch_size) do
        for col = 1, math.ceil(width / patch_size) do
            local hstart = (row - 1) * patch_size + 1
            local wstart = (col - 1) * patch_size + 1
            local hend = math.min(height, hstart+patch_size-1)
            local wend = math.min(width, wstart+patch_size-1)
            hstart = hend - patch_size + 1
            wstart = wend - patch_size + 1

            patch:fill(0)
            patch:copy(img:sub(hstart, hend, wstart, wend))

            local output = cnn:forward(patch:view(1, patch_size, -1)):view(patch_size, -1)
            out_img:sub(hstart, hend, wstart, wend):copy(output:clone())

        end
    end

    out_img:clamp(0, 1)

    for w = 1, width do
        for h = 1, height do
            out_file:write(string.format('%d_%d_%d,%.5f\n', img_number, h, w, out_img[h][w]))
        end
    end
    out_file:flush()

    image.save(path.join(opt.out_dir, i .. '.png'), out_img)
end

out_file:close()
print('compressing submission')
os.execute('rm tmp/submission.csv.gz 1>/dev/null 2>&1')
os.execute('gzip tmp/submission.csv')
