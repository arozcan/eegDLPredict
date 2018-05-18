%save('testresult_17_02_2017.mat');
load('testresult_14_02_2017.mat');
figure;
for i=1:10
    plot(validation_loss(1:end,i));
    hold on
end
plot(mean(validation_loss'),'r','LineWidth',1.5);
xlabel('Egitim k�mesinde iterasyon - Epoch') % x-axis label
ylabel({'Do�rulama Kayb�';'Validation Loss'}) % y-axis label
ylim([0 1.4]);

figure;
plot(mean(validation_loss'));
hold on
plot(mean(train_loss'));
ylim([0 1.4]);

legend('dogrulama kayb�','egitim kayb�')
xlabel('Egitim k�mesinde iterasyon - Epoch') % x-axis label