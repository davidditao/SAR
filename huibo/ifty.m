%%IFFT in row of matrix
function s=ifty(fs)
s=fftshift(ifft(fftshift(fs.'))).';
%s=fftshift(ifft(fftshift(fs),[],2));