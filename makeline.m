clf
table = readtable('data.csv');
table2 = readtable('data2.csv');
table3 = readtable('data3.csv');
table4 = readtable('data4.csv');
table5 = readtable('data5.csv');
table6 = readtable('data6.csv');
table7 = readtable('data7.csv');
table8 = readtable('data8.csv');
table9 = readtable('data9.csv');
table10 = readtable('data10.csv');
table11= readtable('data11.csv');

full = table2array(table);
full2 = table2array(table2);
full3 = table2array(table3);
full4 = table2array(table4);
full5 = table2array(table5);
full6 = table2array(table6);
full7 = table2array(table7);
full8 = table2array(table8);
full9 = table2array(table9);
full10 = table2array(table10);






for row = 3000:3500
    %X = full(row, 1);
    %Y = full(row, 2);
    hold on
    I = imread('bike.png');
    h = image(xlim,-ylim, I);
    uistack(h, 'bottom');
    
    i = row
    
    X2 = full2((row), 1);
    Y2 = full2((row),2);
    
    X3 = full3((row), 1);
    Y3 = full3((row),2);
    
    X4 = full4((row), 1);
    Y4 = full4((row),2);
    
    X5 = full5((row), 1);
    Y5 = full5((row),2);
    
    X6 = full6((row), 1);
    Y6 = full6((row),2);
    
    X7 = full7((row), 1);
    Y7 = full7((row),2);
    
    X8 = full8((row), 1);
    Y8 = full8((row),2);
    
    X9 = full9((row), 1);
    Y9 = full9((row),2);
    
    X10 = full10((row), 1);
    Y10 = full10((row),2);
    
    X = (X2+X3+X3+X4+X5+X6+X7+X8+X9+X10)/10 ;
    Y = (Y2+Y3+Y3+Y4+Y5+Y6+Y7+Y8+Y9+Y10)/10 ;
    
    
    x = 0;
    axis([-1000, 1000, -1000, 1000]);
    scatter(X,Y, 'X')
    scatter(X2,Y2, 'filled')
    scatter(X3,Y3, 'filled')
    scatter(X4,Y4, 'filled')
    scatter(X5,Y5, 'filled')
    scatter(X6,Y6, 'filled')
    scatter(X7,Y7, 'filled')
    scatter(X8,Y8, 'filled')
    scatter(X9,Y9, 'filled')
    scatter(X10,Y10, 'filled')
  
    uistack(h, 'bottom')
    %plot([X2,Y2],[X4,Y4])
    %plot([X3,Y2],[X5,Y5])
    %plot([X3,Y2],[X6,Y6])
   
    %draw_line(X,Y, X2, Y2)
    %draw_line(X2,Y2, X3, Y3)
    %draw_line(X3,Y3, X4, Y4)
    %draw_line(X4,Y4, X5, Y4)
    %raw_line(X5,Y5, X6, Y6)
    pause(.001)
    clf
    hold off
    
    
    
    x = 1;
    
    
    
    
end




function liner = draw_line(v,z,v2,z2)
    
    v7 = v - ((v-v2) *.8) ;
    z7 = z - ((z-z2) *.8);
    
    v3 = v - ((v-v2) *.6) ;
    z3 = z - ((z-z2) *.6);
    
    v4 = v - ((v-v2) *.4) ;
    z4 = z - ((z-z2) *.4);
    
    v5 = v - ((v-v2) *.2) ;
    z5 = z - ((z-z2) *.2);
    
    axis([-1000, 1000, -1000, 1000]);
    scatter(v,z,'filled')
    scatter(v2, z2, 'filled')
    scatter(v3, z3, 'filled')
    scatter(v4, z4, 'filled')
    scatter(v5, z5, 'filled')
    scatter(v7, z7, 'filled')
    
    
end
