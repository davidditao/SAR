function pslr=pslrfunc(x)

l=length(x);
y(1)=x(1);
y(2)=x(l);
i=3;
for k=2:l-1
    if((x(k)-x(k-1))*(x(k)-x(k+1))>0)
        y(i)=x(k);
        i=i+1;
    end
end
fenzhi=max(y);
a=find(y==fenzhi);
y(a)=min(y);
pangban=max(abs(y));
pslr=20*log10(pangban/fenzhi);