%Read the count data files and plot this

clearvars
close all

valDir = './runs/val/';
exps = dir(valDir);
exps = exps(3:end);

disp('Expriment #    |    Mean Err    |     Mean Err %    |    Std Error    |');

for idx = 1:size(exps,1)
   load(strcat(exps(idx).folder,'/',exps(idx).name,'/countData.mat')); 
   
   counterr = detcount-lblcount;
   counterrperc = 100*counterr./lblcount;
   
   edges = -10.5:1:10.5;
   histogram(counterr', edges);
   histname = strcat(exps(idx).folder,'/',exps(idx).name,'/error_hist.png');
   saveas(gcf,histname);

   fprintf('%d %f %f %f \n',idx,mean(counterr),mean(counterrperc),std(cast(counterr,'double')));
   
end