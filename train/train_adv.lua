torch.setdefaulttensortype( 'torch.FloatTensor' )

require 'nn'
require 'lfs'
require 'image'
require 'optim'
require 'randomkit'

-- Commandline options
cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:option( '--batchsize',    16,            'Number of paired data to use in each update.' )
cmd:option( '--roughbatch',   8,             'Number of supervised rough sketches to use in each update.' )
cmd:option( '--linebatch',    8,             'Number of unsupervised line drawings to use in each update.' )
cmd:option( '--patchsize',    384,           'Size of patches to use for training.' )
cmd:option( '--load',         '../model_mse.t7', 'Model to load as a starting point.' )
cmd:option( '--saveinterval', 1000,          'How often to save models (iterations).' )
cmd:option( '--pdata',        'train.csv',   'Supervised dataset.')
cmd:option( '--rdata',        'train_rough.csv', 'Unsupervised rough sketches.')
cmd:option( '--ldata',        'train_line.csv', 'Unsupervised line data.' )
cmd:option( '--threads',       0,             'Number of threads to use (0 is system default0.' )
cmd:option( '--learningRate',  1,            'Learning rate.' )
cmd:option( '--optimizer',    'adadelta',    'Optimizer algorithm.' )
cmd:option( '--scaledata',    '1,1.5,2,2.5,3,3.5,4', 'Dataset scaling (comma separated values).' )
cmd:option( '--thresholdoutput', 0.90,       'Binarization on output.' )
cmd:option( '--thresholdoutputlast', 0.6,    'Binarization on output after rotation.' )
cmd:option( '--kaiten',        180,          'Max degrees to rotate.' )
cmd:option( '--dratio',        8e-5,         'Importance of supervised adversarial loss (alpha).' )
cmd:option( '--uratio',        8e-5,         'Importance of unsupervised adversarial loss (beta).' )
cmd:option( '--pretraindnet',  1000,         'How many iterations to pretrain the discriminator.' )
opt = cmd:parse(arg or {})
print(opt)

require 'cunn'
require 'cudnn'

local model_path = 'cache_adv/'
local imagedir = model_path..'images/'
lfs.mkdir( model_path )
lfs.mkdir( imagedir )

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

function conv3( model, fanin, fanout, stride, padding )
   stride = stride or 1
   padding = padding or 1
   model:add( nn.SpatialConvolution( fanin, fanout, 3, 3, stride, stride, padding, padding ) )
   model:add( nn.ReLU(true) )
   return model
end
model_D = nn.Sequential()
model_D:add( nn.SpatialConvolution( 1, 16, 5, 5, 2, 2, 2, 2 ) ):add( nn.ReLU(true) ) -->1/2
conv3( model_D, 16, 32, 2 ) --> 1/4
conv3( model_D, 32, 64, 2 ) --> 1/8
conv3( model_D, 64, 128, 2 ) --> 1/16
conv3( model_D, 128, 256, 2 ) --> 1/32
model_D:add( nn.Dropout(0.5) )
conv3( model_D, 256, 512, 2 ) --> 1/64
model_D:add( nn.Dropout(0.5) )
model_D:add( nn.Reshape( 512*6*6 ) ):add( nn.Linear( 512*6*6, 1 ) ):add( nn.Sigmoid() )

local lastmodelfile
local iter = 1
if opt.load ~= 'none' then
   lastmodelfile = opt.load
end
for iter = opt.saveinterval,1e10,opt.saveinterval do
   local filename = string.format( '%s/iter%07d.t7', model_path, iter )
   if not file_exists(filename) then break end
   lastmodelfile = filename
   lastiter       = iter
end
if lastmodelfile ~= nil then
   local data = torch.load( lastmodelfile )
   print( 'Loading '..lastmodelfile )
   model_G = data.model
   imgmean = data.mean
   imgstd = data.std
   if data.model_D then
      model_D = data.model_D
   end
   if lastiter then
      iter  = lastiter
   end
end

local criterion_G = nn.MSECriterion()
local criterion_D = nn.BCECriterion()
local real_label = 1
local fake_label = 0

-- Load training data
local paircache = {}
local roughcache = {}
local linecache = {}
--local imgmean = 0
--local imgstd  = 0
local num     = 0
local scales = string.split(opt.scaledata,',')

for line in io.lines( opt.pdata ) do
   local s = line:split(',')
	local rname = s[1]
	local lname = s[2]
	print( 'Loading '..rname..' ...' )
   local inputT  = load_image_gray( rname ):float()
   local outputT = load_image_gray( lname ):float()
   if opt.thresholdoutput > 0 then
      outputT:maskedFill( outputT:lt( opt.thresholdoutput ), 0 )
   end
   for _,s in ipairs(scales) do
      local nh = torch.round( inputT:size(3) / s )
      local nw = torch.round( inputT:size(2) / s )
      local input  = image.scale( inputT,  nh, nw )
      local output = image.scale( outputT, nh, nw )
      local use_image = true
      if opt.patchsize >= math.min(nh, nw) or nh*nw>4000000 then
         use_image = false
      end
      if opt.thresholdoutput > 0 then
         output:maskedFill( output:lt( opt.thresholdoutput ), 0 )
      end
      -- 小さ過ぎたら使えない
      if use_image then
         --imgmean = imgmean + input:mean()
         --imgstd  = imgstd  + input:std()
         num = num + 1
         table.insert( paircache, {input, output} )
      end
   end
end

for line in io.lines( opt.rdata ) do
	local rname = line
	print( 'Loading '..rname..' ...' )
   local inputT  = load_image_gray( rname ):float()
   for _,s in ipairs(scales) do
      local nh = torch.round( inputT:size(3) / s )
      local nw = torch.round( inputT:size(2) / s )
      local input  = image.scale( inputT,  nh, nw )
      local use_image = true
      if opt.patchsize >= math.min(input:size(2), input:size(3)) or nh*nw>4000000 then
         use_image = false
      end
      if use_image then
         --imgmean = imgmean + input:mean()
         --imgstd  = imgstd  + input:std()
         num = num + 1
         table.insert( roughcache, input )
      end
   end
end

for line in io.lines( opt.ldata ) do
	local lname = line
	print( 'Loading '..lname..' ...' )
   local outputT = load_image_gray( lname ):float()
   if opt.thresholdoutput > 0 then
      outputT:maskedFill( outputT:lt( opt.thresholdoutput ), 0 )
   end
   for _,s in ipairs(scales) do
      local nh = torch.round( outputT:size(3) / s )
      local nw = torch.round( outputT:size(2) / s )
      local output = image.scale( outputT, nh, nw )
      local use_image = true
      if opt.patchsize >= math.min(output:size(2), output:size(3)) or nh*nw>4000000 then
         use_image = false
      end
      if opt.thresholdoutput > 0 then
         output:maskedFill( output:lt( opt.thresholdoutput ), 0 )
      end
      if use_image then
         table.insert( linecache, output )
      end
   end
end

for j=1, #paircache do
   paircache[j][1] = (paircache[j][1] - imgmean) / imgstd
end

for j=1, #roughcache do
   roughcache[j] = (roughcache[j] - imgmean) / imgstd
end


function extract_patch( patchsize, input_img, output_img, add_noise )
   local input, output
	add_noise = add_noise or false
   while not input do
      local ur, vr
      local do_rotation = opt.kaiten > 0
      if do_rotation then
         local kaiten = math.min(opt.kaiten,45)*math.pi/180
         local kscale = (math.abs(math.cos(kaiten))
                      + math.abs(math.cos((kaiten-0.5*math.pi))))
         if kscale*patchsize+2 < math.min(output_img:size(2), output_img:size(3)) then
            local border = torch.ceil(((kscale-1)/2)*patchsize)
            ur = randomkit.randint( border+1, output_img:size(2)-patchsize-border-1 )
            vr = randomkit.randint( border+1, output_img:size(3)-patchsize-border-1 )
         else
            do_rotation = false
         end
      end
      if ur==nil or vr==nil then
         ur = randomkit.randint( 1, output_img:size(2)-patchsize )
         vr = randomkit.randint( 1, output_img:size(3)-patchsize )
      end
      local support  = { {}, {ur, ur+patchsize-1}, {vr, vr+patchsize-1} }
      output = output_img[support]
      if output:mean() < 0.99 then
         -- 回転
         if do_rotation then
            local kaiten = randomkit.uniform( -opt.kaiten, opt.kaiten )*math.pi/180
            local kscale = math.abs(math.cos(kaiten)) + math.abs(math.cos((kaiten-0.5*math.pi)))
            local border = torch.ceil(((kscale-1)/2)*patchsize)
            local rsupport = {{}, {support[2][1]-border, support[2][2]+border},
                                  {support[3][1]-border, support[3][2]+border}}
            local csupport = {{}, {border,border+patchsize-1}, {border,border+patchsize-1}}
            output = image.rotate( output_img[rsupport], kaiten, 'bilinear' )[csupport]
            if input_img then
               input = image.rotate( input_img[rsupport],  kaiten, 'bilinear' )[csupport]
            else
               input = true
            end
         else
            if input_img then
               input = input_img[support]
            else
               input = true
            end
         end
      end
   end
   local flipped = randomkit.randint(1,2)==2
   if flipped then
      if input_img then input  = image.hflip( input ) end
      output = image.hflip( output )
   end

	if add_noise then
		local tmp = input:clone()
		if randomkit.randint( 1, 4 ) == 1 then
			local noise = torch.FloatTensor( input:size() )
			local min_n = randomkit.randint(0,3)/10
			local max_n = randomkit.randint(5,7)/10
			if randomkit.randint(1,2) == 1 then
				noise = noise:uniform(min_n, max_n)
			else
				noise = noise:normal(min_n, max_n)
			end
			input = tmp:csub(noise)
		end
	end

   return input, output
end

criterion_G:cuda()
criterion_D:cuda()
model_G:cuda()
model_D:cuda()

local pred        = model_G:forward( torch.randn( 1, 1, opt.patchsize, opt.patchsize ):cuda() )
local outputsize  = pred:size(3)
local outoff = (opt.patchsize-outputsize)/2
if outoff == 0 then outoff = 1 end
local outcrop = {{},{outoff,outoff+outputsize-1},{outoff,outoff+outputsize-1}}

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
local idxlist    = torch.linspace(1,#paircache,#paircache)
local idxlist_r = torch.linspace(1,#roughcache, #roughcache)
local idxlist_l = torch.linspace(1,#linecache, #linecache)
local batch_iter = load_batch_iter( idxlist )
local batch_iter_r = load_batch_iter( idxlist_r )
local batch_iter_l = load_batch_iter( idxlist_l )
local parametersD, gradParametersD = model_D:getParameters()
local parametersG, gradParametersG = model_G:getParameters()
local optparamsD = { learningRate = opt.learningRate }
local optparamsG = { learningRate = opt.learningRate }
print( string.format( '学習開始%d数変数', parametersG:size(1) ) )
while true do
   collectgarbage()
   model_G:training()
   model_D:training()

   sys.tic()

   local Pin      = torch.CudaTensor( opt.batchsize, 1, opt.patchsize, opt.patchsize )
   local Pout     = torch.CudaTensor( opt.batchsize, 1, outputsize,    outputsize )
	local Rin      = torch.CudaTensor( opt.roughbatch, 1, opt.patchsize, opt.patchsize )
	local Lin      = torch.CudaTensor( opt.linebatch, 1, opt.patchsize, opt.patchsize )
   local Y        = torch.CudaTensor( opt.batchsize, 1 )
   local Y_r      = torch.CudaTensor( opt.roughbatch, 1 )
   local Y_l      = torch.CudaTensor( opt.linebatch, 1 )
   local PoutD = torch.CudaTensor( opt.batchsize, 1, outputsize,    outputsize )
   local lossG, lossD
   for j = 1,opt.batchsize do
		-- Paired data
      local idx        = batch_iter()
		local pin_img  = (randomkit.randint(1,10)==10) and (paircache[idx][2]-imgmean) / imgstd or paircache[idx][1]
      local pout_img = paircache[idx][2]
      local pin_patch, pout_patch = extract_patch( opt.patchsize, pin_img, pout_img, true )
      Pin[j]  = pin_patch:cuda()
      Pout[j] = pout_patch[outcrop]:cuda()
      -- Input of the discriminative network
      PoutD[j] = Pout[j]:clone()
	end

	-- Rough sketch data
   for j = 1,opt.roughbatch do
		local idx_r = batch_iter_r()
		local rin_img = roughcache[idx_r]
		local tmp_r, rin_patch = extract_patch( opt.patchsize, nil, rin_img )
		Rin[j] = rin_patch:cuda()
	end

	-- Clean line data
   for j = 1,opt.linebatch do
		local idx_l = batch_iter_l()
		local lin_img = linecache[idx_l]
		local tmp_l, lin_patch = extract_patch( opt.patchsize, nil, lin_img )
		Lin[j] = lin_patch:cuda()
   end

   -- Training
   local ngrad, ngradG, ngradD
   local f_G = function (x)
      gradParametersG:zero() 

      -- Train paired data
      local predG = model_G:forward( Pin ):clone()
      lossG = criterion_G:forward( predG, Pout )
      local gradG= criterion_G:backward( predG, Pout )

		Y:fill(real_label)
      local predD    = model_D:forward( predG )
      criterion_D:forward( predD, Y )
      local cgradD   = criterion_D:backward( predD, Y )
      local gradD    = model_D:backward( predG, cgradD )
      local grad = gradG + gradD * opt.dratio
      model_G:backward( Pin, grad )

      ngrad = grad:norm()
      ngradG = gradG:norm()
      ngradD = gradD:norm() * opt.dratio

		-- Train rough data
		Y_r:fill(real_label)
		local predG_r    = model_G:forward( Rin )
      local predD_r    = model_D:forward( predG_r )
      criterion_D:forward( predD_r, Y_r )
      local cgradD_r   = criterion_D:backward( predD_r, Y_r )
      local gradD_r    = model_D:backward( predG_r, cgradD_r )
		gradD_r = gradD_r*opt.uratio
		model_G:backward( Rin, gradD_r )

      if iter < 1000 or math.fmod( iter, 20 ) == 0 then
         image.save( string.format('%s/%07d.png',  imagedir,iter), predG[1] )
         image.save( string.format('%s/%07d_r.png',imagedir,iter), predG_r[1] )
         image.save( string.format('%s/%07d_ri.png',imagedir,iter), Rin[1]*imgstd+imgmean )
         image.save( string.format('%s/%07d_i.png',imagedir,iter), Pin[1]*imgstd+imgmean )
      end

      return lossG, gradParametersG
   end

   local f_D = function (x)
      gradParametersD:zero() 

      -- Fake image
      local predG = model_G:forward( Pin ):clone()
		Y:fill(fake_label)
      local predD = model_D:forward( predG )
      lossD = criterion_D:forward( predD, Y )
      local cgradD = criterion_D:backward( predD, Y )
      model_D:backward( predG, cgradD )

      -- Real image
		Y:fill(real_label)
      local predD2 = model_D:forward( PoutD )
      lossD = lossD + criterion_D:forward( predD2, Y )
      local cgradD2 = criterion_D:backward( predD2, Y )
      model_D:backward( PoutD, cgradD2 )

		-- Fake image in rough dataset
		local predG_r = model_G:forward( Rin )
		Y_r:fill( fake_label )
		local predD_r = model_D:forward( predG_r )
		lossD = lossD + criterion_D:forward( predD_r, Y_r )
		local cgradD_r = criterion_D:backward( predD_r, Y_r )
      model_D:backward( predG_r, cgradD_r )

      -- Real image in line dataset
		Y_l:fill(real_label)
      local predD_l = model_D:forward( Lin )
      lossD = lossD + criterion_D:forward( predD_l, Y_l )
      local cgradD_l = criterion_D:backward( predD_l, Y_l )
      model_D:backward( Lin, cgradD_l )

		if iter <= opt.pretraindnet and (iter < 1000 or math.fmod( iter, 50 ) == 0) then
			image.save( string.format('%s/%07d.png',imagedir,iter), predG[1] )
			image.save( string.format('%s/%07d_i.png',imagedir,iter), Pin[1]*imgstd+imgmean )
			image.save( string.format('%s/%07d_o.png',imagedir,iter), PoutD[1] )
			image.save( string.format('%s/%07d_r.png',imagedir,iter), predG_r[1] )
			image.save( string.format('%s/%07d_l.png',imagedir,iter), Lin[1] )
		end

      return lossD, gradParametersD
   end

	if iter <= opt.pretraindnet then
		optim[opt.optimizer]( f_D, parametersD, optparamsD )
		print( string.format( '[%07d] Loss=%.2e [%.1f secs]', iter, lossD, sys.toc() ) )
	else
		optim[opt.optimizer]( f_G, parametersG, optparamsG )
		optim[opt.optimizer]( f_D, parametersD, optparamsD )
		print( string.format( '[%07d] Loss=(%.2e, %.2e) Grad=(%.2e, %.2e, %.2e) [%.1f secs]', iter, lossG, lossD, ngrad, ngradG, ngradD, sys.toc() ) )
	end

	iter = iter+1

   if math.fmod( iter, opt.saveinterval ) == 0 then
      local filename = string.format( '%s/iter%07d.t7', model_path, iter )
      print('Saving model to '..filename..'...')
		model_G:clearState()
		model_D:clearState()
		torch.save( filename, { model=model_G, mean=imgmean, std=imgstd, opt=opt, model_D=model_D } )
   end
end





