function plot_shade(tps,YLims,mark)

yy=sort(repmat(YLims,[1,2]),'descend');
for i=1:size(tps,1)
    xx=[tps(i,:),fliplr(tps(i,:))];
    fill(xx,yy,mark,'linestyle','none','FaceColor',[1,0,0.05],'FaceAlpha',0.2);
end

end