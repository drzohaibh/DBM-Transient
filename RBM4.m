trainx = new_data1;
maxepoch =epoch_n(4);
restart=1;
numhid = out_d;
epsilonw      = 0.1;   % Learning rate for weights
epsilonvb     = 0.1;   % Learning rate for biases of visible units    
epsilonhb     = 0.1;   % Learning rate for biases of hidden units     
weightcost  = 0.0002;
initialmomentum  = 0.5;
finalmomentum    = 0.9;

batchsize = 10;
numbatches = length(trainx)/batchsize;
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
        poshidprobs =  (data*vishid) + repmat(hidbiases,numcases,1);                %不一样
        batchposhidprobs(:,:,batch)=poshidprobs;
        posprods    = data' * poshidprobs;
        poshidact   = sum(poshidprobs);
        posvisact = sum(data);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs+randn(numcases,numhid);                           %  高斯
        
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
        neghidprobs = (negdata*vishid) + repmat(hidbiases,numcases,1);              %  不一样
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
% figure
% plot(w,'LineWidth',2)
% set(gca,'box','on')
% xlabel('W','Fontsize',28,'FontWeight','bold');
%%
if out_d == 2
    new_data2 = trainx*vishid + repmat(hidbiases,size(trainx,1),1);
    new_data2 = new_data2+randn(size(trainx,1),numhid); 
    non_sei = size(new_data2, 1) - 100;
    figure
    scatter(new_data2(1:non_sei,1), new_data2(1:non_sei,2),'xb','LineWidth',3)
    hold on
    scatter(new_data2(non_sei+1:size(new_data2, 1),1), new_data2(non_sei+1:size(new_data2, 1),2),'*r','LineWidth',1.2)
    set(gca,'box','on')
    title([state, ' ', c],'FontSize',28,'FontWeight','bold')
    set(gca,'xtick',[],'xticklabel',[])
    set(gca,'ytick',[],'yticklabel',[])
    xlabel('feature one','FontSize',28,'FontWeight','bold');
    ylabel('feature two','FontSize',28,'FontWeight','bold');
end