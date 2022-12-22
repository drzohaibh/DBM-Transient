all_x = [];
all_label = [];
 for i = 1:19
     load(['c301_raw_data/raw_data/non_',num2str(i),'.mat' ])
     load(['c301_raw_data/label/label_',num2str(i),'.mat' ])
     z = x(:,1);
     all_x = [all_x; x(:,1)];
     all_label = [all_label; label];
 end
 
 all_non_sei = [];
 all_sei = [];
 for i = 1:length(all_label)
     if all_label(i) == 0
         all_non_sei = [all_non_sei; all_x(i)];
     else
         all_sei = [all_sei; all_x(i)];
    end
end
 
sample_non = [];
sample_sei = [];
 
for i = 1:floor(length(all_non_sei)/512)-1
     sample_non = [sample_non; all_non_sei((1+512*(i-1)):(512+512*(i)))];
end
 
for i = 1:floor(length(all_sei)/512)-1
     sample_sei = [sample_sei; all_sei((1+512*(i-1)):(512+512*(i)))];
end
