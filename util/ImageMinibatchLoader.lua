
import 'image'
import 'util.misc'
import 'util.ZCA'

local ImageMinibatchLoader = {}
ImageMinibatchLoader.__index = ImageMinibatchLoader

-- split_index is integer: 1 = train, 2 = val, 3 = test

PREPRO_TABLE_THRESHOLD = 10000

function ImageMinibatchLoader.create(opt)
    data_dir = opt.data_dir
    label_dir = path.join(data_dir, 'y')
    data_dir = path.join(data_dir, 'x')
    test_dir = opt.test_dir
    torch_dir = opt.torch_dir

    local self = {}
    setmetatable(self, ImageMinibatchLoader)

    -- TODO: make dependent on opts
    self.apply_zca = opt.apply_zca
    self.val_part = opt.val_part
    self.patches_per_file = opt.patches_per_file
    self.patch_size = opt.patch_size
    self.batch_size = opt.batch_size
    self.x_torch_prefix = path.join(torch_dir, 'data')
    self.y_torch_prefix = path.join(torch_dir, 'label')
    self.torch_dir = torch_dir
    self.batch_size = opt.batch_size

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if (not path.exists(self.x_torch_prefix .. '_train1.t7') or not path.exists(self.y_torch_prefix .. '_train1.t7')) then
        -- prepro files do not exist, generate them
        print('data1.t7 or label1.t7 doesn\'t exist. Running preprocessing...')
        run_prepro = true
    end

    if run_prepro then
        -- preprocess files and save the tensors
        print('one-time setup: preprocessing input...')
        local data_files, val_files, test_files =
                self:glob_raw_data_files(data_dir, label_dir, test_dir)
        print('parsing training data')
        self.total_samples = self:preprocess(data_files,
                                             self.x_torch_prefix .. '_train',
                                             self.y_torch_prefix .. '_train')
        torch.save(path.join(torch_dir, 'sample_count.t7'), self.total_samples)
        print('parsing validation data')
        self:preprocess(val_files,
                        self.x_torch_prefix .. '_val',
                        self.y_torch_prefix .. '_val')
        print('parsing test data')
        self:preprocessTest(test_files,
                      self.x_torch_prefix .. '_test')
    end
    self.zca = nil

    local train_ct, val_ct, test_ct = self.count_prepro_files(torch_dir)
    self.file_count = {train_ct, val_ct, test_ct}
    self.data_loaded = {false, false, false}
    self.total_samples = torch.load(path.join(torch_dir, 'sample_count.t7'))
    self.batch_idx = {0, 0, 0}
    self.file_idx = {1, 1, 1}

    self:load_file(1, 1)

    print('data load done.')
    collectgarbage()
    return self
end

function ImageMinibatchLoader:load_file(split_index, index)
    if split_index < 1 or split_index > 3 then
        printRed('invalid split index in load_file: ' .. split_index)
        os.exit()
    end
    if self.file_count[split_index] < index then
        printRed('invalid file index in load_file: split=' .. split_index .. ', index=' .. index)
        os.exit()
    end

    local split_names = {'_train', '_val', '_test'}
    local modifier = split_names[split_index]

    local x_path = self.x_torch_prefix .. modifier .. index .. '.t7'
    local y_path = self.y_torch_prefix .. modifier .. index .. '.t7'
    if (not path.exists(x_path) or not path.exists(y_path)) then
        printRed('trying to load inexistent files! (' .. x_path .. ', ' .. y_path .. ')')
        os.exit()
    end

    print('loading data part ' .. index .. ' from split ' .. split_index)
    local data = torch.load(x_path)
    local labels = torch.load(y_path)
    local len = #data

    self.x_batches = data
    self.y_batches = labels

    self.data_loaded = {false, false, false}
    self.data_loaded[split_index] = true
    self.file_idx[split_index] = index
    self.batch_idx[split_index] = 0

    return true
end

function ImageMinibatchLoader:reset_batch_pointer(split_index, batch_index, file_index)
    batch_index = batch_index or 0
    file_index = file_index or 1
    self.batch_idx[split_index] = batch_index
    self.file_idx[split_index] = file_index
end

-- TODO: add support for other sets
function ImageMinibatchLoader:refresh()
    local prev_batch_idx = self.batch_idx[1]
    self:load_file(1, self.file_idx[1])
    self.batch_idx[1] = prev_batch_idx
    print('resuming from batch ' .. prev_batch_idx)
end

function ImageMinibatchLoader:next_batch(split_index)
    -- load data
    if not self.data_loaded[split_index] then
        local prev_batch_idx = self.batch_idx[split_index]
        self:load_file(split_index, self.file_idx[split_index])
        self.batch_idx[split_index] = prev_batch_idx
        if prev_batch_idx > 0 then
            print('resuming from batch ' .. prev_batch_idx)
        end
    end

    self.batch_idx[split_index] = self.batch_idx[split_index] + 1
    if self.batch_idx[split_index] > #self.x_batches then
        -- load next file
        local file_idx = self.file_idx[split_index] + 1
        if file_idx > self.file_count[split_index] then
            file_idx = 1 -- wrap around file count
        end
        self:load_file(split_index, file_idx) -- sets new file index
        self.batch_idx[split_index] = 1
    end

    local final_batch_idx = self.batch_idx[split_index]
    return self.x_batches[final_batch_idx], self.y_batches[final_batch_idx]
end

-- *** STATIC methods ***
function ImageMinibatchLoader.count_prepro_files(prepro_dir)
    local train = 0
    local val = 0
    local test = 0

    for file in lfs.dir(prepro_dir) do
        if file:find('data_train') then
            train = train + 1
        elseif file:find('data_val') then
            val = val + 1
        elseif file:find('data_test') then
            test = test + 1
        end

    end

    return train, val, test
end


function ImageMinibatchLoader:glob_raw_data_files(data_dir, label_dir, test_dir)
    local data = {}
    local labels = {}
    local raw_data = {}
    local raw_labels = {}
    local val_data = {}
    local val_labels = {}
    local test_files = {}

    for file in lfs.dir(data_dir) do
        if file:find('.png') then
            table.insert(raw_data, path.join(data_dir, file))
        end
    end

    for file in lfs.dir(label_dir) do
        if file:find('.png') then
            table.insert(raw_labels, path.join(label_dir, file))
        end
    end

    for file in lfs.dir(test_dir) do
        if file:find('.png') then
            table.insert(test_files, path.join(test_dir, file))
        end
    end

    index_table = {}
    index_count = 0
    while index_count < math.floor(#raw_data * self.val_part) do
        index = (torch.random() % #raw_data) + 1
        if not index_table[index] then
            index_table[index] = true
            index_count = index_count + 1
        end
    end

    for i = 1, #raw_data do
        if not index_table[i] then
            table.insert(data, raw_data[i])
            table.insert(labels, raw_labels[i])
        else
            table.insert(val_data, raw_data[i])
            table.insert(val_labels, raw_labels[i])
        end
    end

    assert(#data == #labels)
    assert(#val_data == #val_labels)

    printBlue(string.format('found %d training files, %d validation files', #data, #val_data))

    return {data=data, labels=labels}, {data=val_data, labels=val_labels}, test_files
end

function ImageMinibatchLoader:preprocessTest(test_files, filename)
    local test_images = {}
    table.sort(test_files)
    for i, file in ipairs(test_files) do
        xlua.progress(i, #test_files)
        local file_number = tonumber(path.basename(file):sub(0, -5))
        local img = image.load(file)
        local height = img:size(2)
        local width = img:size(3)

        if self.apply_zca then
            local transformed_img = torch.Tensor(height, width)
            local patch = torch.Tensor(self.patch_size, self.patch_size)
            for row = 1, math.ceil(height / self.patch_size) do
                for col = 1, math.ceil(width / self.patch_size) do
                    local hstart = (row - 1) * self.patch_size + 1
                    local wstart = (col - 1) * self.patch_size + 1
                    local hend = math.min(height, hstart+self.patch_size-1)
                    local wend = math.min(width, wstart+self.patch_size-1)
                    hstart = hend - self.patch_size + 1
                    wstart = wend - self.patch_size + 1

                    patch:copy(img:sub(1, 1, hstart, hend, wstart, wend))

                    local output = self.zca:transform(patch:view(1, -1)):view(self.patch_size, -1)

                    transformed_img:sub(hstart, hend, wstart, wend):copy(output)
                end
            end
            img = transformed_img:view(1, height, width)
        end

        table.insert(test_images, {img_number=file_number, img_data=img})
    end
    torch.save(filename .. '1.t7', test_images)
end

function ImageMinibatchLoader:preprocess(input_files, input_filename, label_filename)
    local data_table = {}
    local label_table = {}
    local timer = torch.Timer()

    local data_idx = 1


    -- helper function
    function saveTensors()
        assert(#data_table == #label_table)

        if self.apply_zca and not self.zca then
            self.zca = ZCA()
            local zca_data = torch.Tensor(#data_table, self.patch_size * self.patch_size)
            for i = 1, #data_table do
                zca_data[i]:copy(data_table[i]:view(-1))
            end
            self.zca:fit(zca_data)
        end

        collectgarbage()

        for i = 1, #data_table do
            data_table[i] = self.zca:transform(data_table[i]:view(1, -1)):view(1, self.patch_size, -1)
        end

        data_batch_table = {}
        label_batch_table = {}
        for i = 1, math.floor(#data_table / self.batch_size), self.batch_size do
            data_batch = {}
            label_batch = {}
            for j = i, i+self.batch_size-1 do
                table.insert(data_batch, data_table[j])
                table.insert(label_batch, label_table[j])
            end
            table.insert(data_batch_table, data_batch)
            table.insert(label_batch_table, label_batch)
        end
        data_table = nil
        label_table = nil
        collectgarbage()

        torch.save(input_filename .. data_idx .. '.t7', data_batch_table)
        torch.save(label_filename .. data_idx .. '.t7', label_batch_table)

        -- empty temporary tables
        data_idx = data_idx + 1
        data_table = {}
        label_table = {}
        collectgarbage()
    end



    for i = 1, #input_files.data do

        xlua.progress(i, #input_files.data)

        local img_x = image.load(input_files.data[i])
        local img_y = image.load(input_files.labels[i])
        -- image is loaded with dims (1, height, width)
        img_x = img_x:view(img_x:size(2), -1)
        img_y = img_y:view(img_y:size(2), -1)

        local height = img_x:size(1)
        local width = img_x:size(2)
        local hrange = height - self.patch_size
        local wrange = width - self.patch_size

        for i = 1,self.patches_per_file do
            local hstart = (torch.random() % hrange) + 1
            local wstart = (torch.random() % wrange) + 1
            local patch_x = image.crop(img_x, wstart, hstart,
                                       wstart+self.patch_size, hstart+self.patch_size):view(self.patch_size, -1)
            local patch_y = image.crop(img_y, wstart, hstart,
                                       wstart+self.patch_size, hstart+self.patch_size):view(self.patch_size, -1)
            table.insert(data_table, patch_x)
            table.insert(label_table, patch_y)

            ----------------- save tensors -----------------
            -- save tensors if tables are growing big
            -- for lua's 2GB memory limit 1e6 entries is already big
            if #data_table > PREPRO_TABLE_THRESHOLD then
                saveTensors()
            end
        end
    end


    if #data_table > 0 then
        saveTensors()
    end
    printYellow(string.format("Elapsed: %.2fs", timer:time().real))

    collectgarbage()
    return #input_files.data * self.patches_per_file
end

return ImageMinibatchLoader
