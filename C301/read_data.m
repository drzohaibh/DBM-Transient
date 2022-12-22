All_non = [];
All_sei = [];

for i = 1:19

    load(['C301_data/c301_raw_data/raw_data/non_',num2str(i),'.mat'])
    load(['C301_data/c301_raw_data/label/label_',num2str(i),'.mat'])
    
    fs = 256; 
    t  = (1:size(label))'./fs;
    [tps_sei,tps_non]= get_tps(t,label,fs);
    
    sei_dur = tps_sei(:,2) - tps_sei(:,1);
    All_sei = [All_sei; sei_dur];
    
    non_dur = tps_non(:,2) - tps_non(:,1);
    All_non = [All_non; non_dur];

    YLims = [min(x(:,1)),max(x(:,1))];
    hfig  = figure();

    plot_shade(tps_sei,YLims,'c-'); hold on;
    plot_shade(tps_non,YLims,'r-'); 
    plot(t, x(:,1));

end    
    %% Check whether the boundary tps are correctly picked up

    YLims = [min(x(:,1)),max(x(:,1))];
    hfig  = figure();
    subplot(1,2,1)
    plot_shade(tps_sei,YLims,'c-'); hold on;
    plot_shade(tps_non,YLims,'r-'); 
    plot(t, x(:,1));

    
    %% Plotting Histogram
    hfig  = figure();
    subplot(1,2,1)
    histogram(All_sei,'BinWidth',4);
    title(['Sei'])
%     xlim([0 10]) 

    subplot(1,2,2)
    histogram(All_non,'BinWidth',4);
%     xlim([0 10]) 

    title(['Non-Sei'])

   
% exportgraphics(gcf,'All-Subjects.png','Resolution', 400)



