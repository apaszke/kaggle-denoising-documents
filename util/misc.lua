function sliceTable(tab, to)
    local slice = {}
    for i = 1, to do
        slice[i] = tab[i]
    end
    return slice
end

function calculate_avg_loss(losses)
    local smoothing = 40
    local sum = 0
    for i = #losses, math.max(1, #losses - smoothing + 1), -1 do
        sum = sum + losses[i]
    end
    return sum / math.min(smoothing, #losses)
end

function mergeTensorTable(tab)
    if #tab == 1 then
      return tab[1]
    end
    local dims = tab[1]:size():size(1)
    local size = {[1] = 1}
    for j=1,dims do
      size[j+1] = tab[1]:size(j)
    end
    print(tab)
    local merged = tab[1]:view(table.unpack(size))
    for i=2,#tab do
      assert(tab[i]:size():size() == dims)
      for j = 1,dims do
        size[j+1] = tab[i]:size(j)
      end
      merged = torch.cat(merged, tab[i]:view(table.unpack(size)), 1)
    end

    return merged
end
