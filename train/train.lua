require 'nn'
require 'lfs'
require 'image'
require 'optim'
require 'randomkit'

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:option( '--batchsize',    8,             'Number of patches to use in each batch.' )
cmd:option( '--batchgroups',  1,             'Batches to accumulate before updating the weights.')
cmd:option( '--patchnum',     1,             'Number of patches to extract from each image.' )
cmd:option( '--patchsize',    424,           'Size of the patches to use (in pixels).' )
cmd:option( '--load',         'none',        'Name of the file to load and continue training. \'none\' defaults to training from scratch.' )
cmd:option( '--saveinterval', 2500,          'Number of iterations between each save.' )
cmd:option( '--trainfiles',   'train.csv',   'Training file.' )
cmd:option( '--scaledata',    '1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5',     'Different scales to use when training (comma-separated list).' )
cmd:option( '--kaiten',       180,            'Degrees to randomly rotate when training.' )
opt = cmd:parse(arg or {})
print(opt)

require 'cunn'
require 'cudnn'

local use_weight = true

local function conv1( model, fin, fout ) 
   return model:add( nn.SpatialConvolution( fin, fout, 3, 3, 1, 1, 1, 1 ) )
end
local function conv3( model, fin, fout, step, no_relu ) 
   step = step or 1
	no_relu = no_relu or false
   model:add( nn.SpatialConvolution( fin, fout, 3, 3, step, step, 1, 1 ) )
   if not no_relu then
	   model:add( nn.SpatialBatchNormalization( fout ) )	
      model:add( nn.ReLU(true) )
   end
   return model
end
local function conv5( model, fin, fout, step, no_relu ) 
   step = step or 1
   do_relu = do_relu or true
   model:add( nn.SpatialConvolution( fin, fout, 5, 5, step, step, 2, 2 ) )
   if not no_relu then
      model:add( nn.SpatialBatchNormalization( fout ) )
      model:add( nn.ReLU(true) )
   end
   return model
end
local function deconv2( model, fin, fout )
   model:add( nn.SpatialFullConvolution( fin, fout, 4, 4, 2, 2, 1, 1 ) )
	model:add( nn.SpatialBatchNormalization( fout ) )	
	model:add( nn.ReLU(true) )
	return model
end
local function bn( fin )
   return nn.SpatialBatchNormalization( fin )
end
local model = nn.Sequential()
conv5( model, 1, 48, 2 ) --> 1/2
conv3( model, 48, 128 )
conv3( model, 128, 128 )
conv3( model, 128, 256, 2 ) --> 1/4
conv3( model, 256, 256 )
conv3( model, 256, 256 )
conv3( model, 256, 256, 2 ) --> 1/8
conv3( model, 256, 512 )
conv3( model, 512, 1024 )
conv3( model, 1024, 1024 )
conv3( model, 1024, 1024 )
conv3( model, 1024, 1024 )
conv3( model, 1024, 512 )
conv3( model, 512, 256 )
deconv2( model, 256, 256 ) --> 1/4
conv3( model, 256, 256 )
conv3( model, 256, 128 )
deconv2( model, 128, 128 ) --> 1/2
conv3( model, 128, 128 )
conv3( model, 128, 48 )
deconv2( model, 48, 48 ) --> 1
conv3( model, 48, 24 )
conv3( model, 24, 1, 1, true ):add( nn.Sigmoid(true) )

savedir = 'cache/'
imagedir = savedir..'/images/'
lfs.mkdir( savedir )
lfs.mkdir( imagedir )

torch.setdefaulttensortype( 'torch.FloatTensor' )

---------  Functions  -----------
--Note: 'rotdeg' is degree, not radian
local function rotImage( img, rotdeg )
   rotdeg = rotdeg or 0
   local rot_angle = rotdeg/180*math.pi
   local out = image.rotate(img, rot_angle, 'bilinear')
   return out
end
function load_image_gray ( path )
   local img = image.load( path )
   if img:size(1) > 3 then -- rgb + alpha
      img = img[{{1,3},{},{}}]
   end
   if img:size(1) == 2 then -- grayscale + alpha
      img = img[{{1},{},{}}]
   end
   if img:size(1) > 1 then -- rgb -> grayscale
      img = image.rgb2y( img )
   end
   return img
end
function file_exists(name)
   local f=io.open(name,'r')
   if f~=nil then io.close(f) return true else return false end
end
------------------------------------

local lastmodelfile
local iter = 0 
if opt.load ~= 'none' then
   lastmodelfile = opt.load
end
for iter = opt.saveinterval,1e10,opt.saveinterval do
   local filename = string.format( '%s/%s/iter%07d.t7', models_patch, opt.savename, iter )
   if not file_exists(filename) then break end
   lastmodelfile = filename
   lastiter       = iter
end
if lastmodelfile ~= nil then
   print( lastmodelfile..'の中からモデルを読み込み中…' )
   local data = torch.load( lastmodelfile )
   if torch.typename(data)=='nn.Sequential' then
      model = data
   else
      model = data.model
   end
   if lastiter then
      iter  = lastiter
   end
end

local criterion = nn.MSECriterion()

function string:split(sep)
   local sep, fields = sep or ":", {}
   local pattern = string.format("([^%s]+)", sep)
   self:gsub(pattern, function(c) fields[#fields+1] = c end)
   return fields
end   

local dataset = {}
for line in io.lines( opt.trainfiles ) do
   local s = line:split(',')
   dataset[ s[1] ] = s[2]
end


print( model )

function compute_weightmap( img, nbins, dist )
   nbins = nbins or 10
   dist  = dist  or 4
   local lin = torch.linspace( 0, 1, nbins+1 )
   local out = torch.zeros( img:size() )

   for x = 1,img:size(2) do
      local xmin = math.max(           1, x-dist)
      local xmax = math.min( img:size(2), x+dist)
      for y = 1,img:size(3) do
         local ymin = math.max(           1, y-dist)
         local ymax = math.min( img:size(3), y+dist)
         local I    = img[{{},{xmin,xmax},{ymin,ymax}}]
         local bins = {}
         for i = 1,nbins do
            bins[i] = (I:ge( lin[i] ):cmul( I:le( lin[i+1] ))):sum() / (I:size(2)*I:size(3))
         end
         local p = img[1][x][y]
         local n
         for i = 1,nbins do
            if p >= lin[i] and p <= lin[i+1] then
               n = i
               break
            end
         end
         out[1][x][y] = math.exp( -bins[n] ) + 0.5
      end
   end
   return out
end

local imgcache = {}
local wcache = {}
local imgmean = 0
local imgstd  = 0
local num     = 0
local scales = string.split(opt.scaledata,',')
for k,v in pairs(dataset) do
   print('   '..k)
   local inputT  = load_image_gray( k ):float()
   local outputT = load_image_gray( dataset[k] ):float()
   outputT:maskedFill( outputT:lt( 0.9 ), 0 )
   for _,s in ipairs(scales) do
      local nh = torch.round( inputT:size(3) / s )
      local nw = torch.round( inputT:size(2) / s )
      local input  = image.scale( inputT,  nh, nw )
      local output = image.scale( outputT, nh, nw )
      local use_image = true
      if opt.patchsize > math.min(input:size(2), input:size(3)) then
         use_image = false
      end
      output:maskedFill( output:lt( 0.9 ), 0 )
      if use_image and use_weight then
         local w
         local wfile = v:split('/')
         lfs.mkdir('wcache')
         wfile = string.format('wcache/hist_local_s%d_th%d_%s',s*100,90,wfile[#wfile]:sub(1,-5))
         if file_exists(wfile..'.t7') then
            w = torch.load(wfile..'.t7')
         else
            print('   '..wfile..'を計算中…')
            w = compute_weightmap( output, 10, 2 )
            torch.save(wfile..'.t7',w)
            image.save(wfile..'.png',w/w:max())
         end
         table.insert( wcache, w )
      end
      if use_image then
         imgmean = imgmean + input:mean()
         imgstd  = imgstd  + input:std()
         num = num+1
         table.insert( imgcache, {input, output} )
      end
   end
end
imgmean = imgmean / num
imgstd  = imgstd / num
for j=1, #imgcache do
   imgcache[j][1] = (imgcache[j][1] - imgmean) / imgstd
end

function extract_patch( patchsize, input_img, output_img, weight_img )
   local input, output, weight
   while not input do
      local ur, vr
      local kaitensuru = opt.kaiten > 0
      if kaitensuru then
         local kaiten = math.min(opt.kaiten,45)*math.pi/180
         local kscale = (math.abs(math.cos(kaiten))
                      + math.abs(math.cos((kaiten-0.5*math.pi))))
         if kscale*patchsize+2 < math.min(input_img:size(2), input_img:size(3)) then
            local border = torch.ceil(((kscale-1)/2)*patchsize)
            ur = randomkit.randint( border+1, input_img:size(2)-patchsize-border-1 )
            vr = randomkit.randint( border+1, input_img:size(3)-patchsize-border-1 )
         else
            kaitensuru = false
         end
      end
      if ur==nil or vr==nil then
         ur = randomkit.randint( 1, input_img:size(2)-patchsize )
         vr = randomkit.randint( 1, input_img:size(3)-patchsize )
      end
      local support  = { {}, {ur, ur+patchsize-1}, {vr, vr+patchsize-1} }
      output = output_img[support]
      if output:mean() < 0.99 then
         if kaitensuru then
            local kaiten = randomkit.uniform( -opt.kaiten, opt.kaiten )*math.pi/180
            local kscale = math.abs(math.cos(kaiten)) + math.abs(math.cos((kaiten-0.5*math.pi)))
            local border = torch.ceil(((kscale-1)/2)*patchsize)
            local rsupport = {{}, {support[2][1]-border, support[2][2]+border},
                                  {support[3][1]-border, support[3][2]+border}}
            local csupport = {{}, {border,border+patchsize-1}, {border,border+patchsize-1}}
            output = image.rotate( output_img[rsupport], kaiten, 'bilinear' )[csupport]
            input  = image.rotate( input_img[rsupport],  kaiten, 'bilinear' )[csupport]
            if weight_img then
               weight = image.rotate( weight_img[rsupport], kaiten, 'bilinear' )[csupport]
            end
         else
            input  = input_img[support]
            if weight_img then
               weight = weight_img[support]
            end
         end
      end
   end
   local flipped = randomkit.randint(1,2)==2
   if flipped then
      input  = image.hflip( input )
      output = image.hflip( output )
      if weight_img then
         weight = image.hflip(weight)
      end
   end
   output:maskedFill( output:lt( 0.6 ), 0 )
   return input, output, weight
end


criterion:cuda()
model:cuda()
local pred        = model:forward( torch.randn( 1, 1, opt.patchsize, opt.patchsize ):cuda() )
local outputsize  = pred:size(3)

local smallest = 1e100
for k,v in ipairs(imgcache) do
   local s = v[1]:size(2)*v[1]:size(3)
   if s < smallest then
      smallest = s
   end
end
local idlist = {}
for k,v in ipairs(imgcache) do
   local s = v[1]:size(2)*v[1]:size(3)
   local n = torch.round(s / smallest)
   for i = 1,n do
      idlist[ #idlist+1 ] = k
   end
end


function load_batch_iter( data )
   epoch = epoch or 1
   local n
   if torch.typename(data) then
      n = data:size(1)
   else
      n = #data
   end
   local i = 1
   local r = torch.randperm(n)
   return function ()
         i = i + 1
         if i > n then
            i = 1
            r = torch.randperm(n)
            epoch = epoch+1
         end
         return data[ r[i] ]
      end
end
local batch_iter = load_batch_iter( idlist )
local parameters, gradParameters = model:getParameters()
local optparams = {
   learningRate = opt.learningRate
}
local batchsize = opt.batchsize
print( string.format( '学習開始%d数変数', parameters:size(1) ) )
while true do
   collectgarbage()
   model:training()

   local feval = function (x)
      sys.tic()
      gradParameters:zero()
      
      local loss   = 0
      local input  = torch.CudaTensor( opt.batchsize*opt.patchnum, 1, opt.patchsize, opt.patchsize )
      local output = torch.CudaTensor( opt.batchsize*opt.patchnum, 1, outputsize,    outputsize )
      local weight
      if use_weight then
         weight = torch.CudaTensor( opt.batchsize*opt.patchnum, 1, outputsize, outputsize )
      end
      for i = 1,opt.batchgroups do
         for j = 1,opt.batchsize do 
            local idx        = batch_iter()
            local input_img  = imgcache[ idx ][1]
            local output_img = imgcache[ idx ][2]
            local weight_img = wcache[ idx ]
            for p = 1,opt.patchnum do
               local inputp, outputp, weightp = extract_patch( opt.patchsize,
                     input_img, output_img, weight_img )
               local n = opt.patchnum*(j-1)+p
               local outoff = (opt.patchsize-outputsize)/2
               if outoff == 0 then outoff = 1 end
               local outcrop = {{},{outoff,outoff+outputsize-1},{outoff,outoff+outputsize-1}}
               input[n]  = inputp:cuda()
               output[n] = outputp[outcrop]:cuda()
               if use_weight then
                  weight[n] = weightp[outcrop]:cuda()
               end
            end
         end
         if use_weight then
            criterion.weight = weight:cuda()
         end

         local pred  = model:forward( input )
         loss = loss + criterion:forward(  pred, output ) 
         local grad  = criterion:backward( pred, output )
         model:backward( input, grad )

         if i==1 and (iter < 1000 or math.fmod(iter, 20) == 0) then
            image.save( string.format('%s/%07d.png',imagedir,iter),   pred[1] )
            image.save( string.format('%s/%07d_i.png',imagedir,iter), input[1]*imgstd + imgmean )
            image.save( string.format('%s/%07d_o.png',imagedir,iter), output[1] )
         end
      end
      gradParameters:div( opt.batchgroups )
      loss = loss / opt.batchgroups
      print( string.format( '[%07d] Loss= %.2e [%.1f secs]', iter, loss, sys.toc() ) )
      return loss, gradParameters
   end

   optim['adadelta']( feval, parameters, optparams )
   iter = iter+1

   if math.fmod( iter, opt.saveinterval ) == 0 then
      local filename = string.format( '%s/iter%07d.t7', savedir, iter )
      print('Saving model to '..filename..'...')
      model:clearState()
      torch.save( filename, { model=model, mean=imgmean, std=imgstd, opt=opt } )
   end
end
