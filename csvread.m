clf
table = readtable('data.csv');
full = table2array(table);

hold on

for row = 1:214
    X = full(row, 1);
    Y = full(row, 2);
    
    
    
    X2 = full((row+2), 1);
    Y2 = full((row+2),2);
    
    
    
    axis([-500, 500, -500, 500]);
    scatter(X,Y,'filled')
    
    
    scatter(X2, Y2, 'filled')
    pause(.001)
    
    
end
hold off
