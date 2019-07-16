lena512= imread('lena512.bmp');
histogram(lena512);
%imhist(lena512);
title('Histogram of lena image');
xlabel('Gray Level');
ylabel('Pixel Count');
