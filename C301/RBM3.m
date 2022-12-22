trainx = new_data3;
maxepoch = epoch_n(3);      %1
restart=1;
numhid = 25;
epsilonw      = 0.1;   % Learning rate for weights
epsilonvb     = 0.1;   % Learning rate for biases of visible units    
epsilonhb     = 0.1;   % Learning rate for biases of hidden units     
weightcost  = 0.0002;
initialmomentum  = 0.5;
finalmomentum    = 0.9;

batchsize = 10;
numbatches = floor(length(trainx)/batchsize);
batchdata = zeros(batchsize,size(trainx,2),numbatches);
for i=1:numbatches
    batchdata(:,:,i) = trainx((1+(i-1)*batchsize):batchsize*i, :);
end
[numcases,numdims,numbatches]=size(batchdata);
if restart ==1
    restart=0;
    epoch=1;
    % Initializing symmetric weights and biases.
    rng(2)
    vishid     = 0.1*randn(numdims, numhid);
    hidbiases  = zeros(1,numhid);
    visbiases  = zeros(1,numdims);
    
    poshidprobs = zeros(numcases,numhid);
    neghidprobs = zeros(numcases,numhid);
    posprods    = zeros(numdims,numhid);
    negprods    = zeros(numdims,numhid);
    
    vishidinc  = zeros(numdims,numhid);
    hidbiasinc = zeros(1,numhid);
    visbiasinc = zeros(1,numdims);
    batchposhidprobs=zeros(numcases,numhid,numbatches);
end
w_coll = [];
for epoch = epoch:maxepoch                %1:50
    fprintf(1,'epoch %d\r',epoch);
    errsum=0;
    for batch = 1:numbatches
        fprintf(1,'epoch %d batch %d\r',epoch,batch);
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);
        poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));
        batchposhidprobs(:,:,batch)=poshidprobs;
        posprods    = data' * poshidprobs;
        poshidact   = sum(poshidprobs);                  
        posvisact = sum(data);                          
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);   
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
        neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));
        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata);
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum((data-negdata).^2 ));
        errsum = err + errsum;
        if epoch>5
            momentum=finalmomentum;
        else
            momentum=initialmomentum;
        end
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = momentum*vishidinc + ...
            epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
        visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
    w_coll = [w_coll;vishid];
end
w = [];
for i = 1:maxepoch
    w =[w;w_coll(((i-1)*numdims+1),1)];
end

new_data1 = 1./(1 + exp(-trainx*vishid - repmat(hidbiases,size(trainx,1),1)));
new_data1 = mapminmax(new_data1')';
new_data1 = (new_data1 > rand(size(new_data1,1),size(new_data1,2)))*1;
% figure;imagesc(new_data1)