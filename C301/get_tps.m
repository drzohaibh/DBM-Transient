function [tps_sei,tps_non]=get_tps(t,label,fs)

tps=t(1)+find(diff(label))./fs;

if numel(tps)<2
    error('tps is empty');
else
    if mod(numel(tps),2)==1
        mid=floor((tps(1)+tps(2))*fs/2);
        if label(mid)==1
            tps=[tps;t(end)];
        else
            tps=[t(1);tps];
        end
    end
end

tps_sei=reshape(tps,[2, numel(tps)/2]);
tps_sei=tps_sei';

%% To get nonseizure start and end time
if tps(1)~= t(1)
    tps = [t(1);tps];
else
    tps(1) =[];
end

if tps(end)~=t(end)
    tps = [tps;t(end)];
else
    tps(end) = [];
end

tps_non=reshape(tps,[2,numel(tps)/2]);
tps_non=tps_non';

end