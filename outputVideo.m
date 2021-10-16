clf
table = readtable('data.csv');
full = table2array(table)

%myVideo = VideoWriter('myVideoFile', 'MPEG-4'); %open video file
%myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
%open(myVideo)

h = figure;
axis tight manual

hold on
filename = 'GIFtest'



for row = 1:214
    X = full(row, 1);
    Y = full(row, 2);
    axis([-500, 500, -500, 500])
    plot(X,Y,'o')
    pause(.01)
    
    %frame = getframe(gcf); %get frame
    %writeVideo(myVideo, frame);
    
    frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
    
    if row == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
end
hold off
