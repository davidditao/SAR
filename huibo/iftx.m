%%IFFT in column of matrix
function s=iftx(fs)
s=fftshift(ifft(fftshift(fs)));
%s=fftshift(ifft(fftshift(fs),[],1));